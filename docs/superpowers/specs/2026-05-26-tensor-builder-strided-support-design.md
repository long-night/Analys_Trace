# TensorBuilder 非连续张量支持设计文档

> **日期**: 2026-05-26
> **状态**: 已确认
> **主题**: 修复 `_build_tensor` 无法正确处理大 stride、负 stride、广播等边界情况

---

## 1. 背景与问题分析

### 1.1 当前实现

`TensorBuilder._build_tensor()` 采用两步法创建张量：

```python
def _build_tensor(self, dims, strides, dtype, op_name=""):
    # Step 1: 创建连续张量（大小 = product(dims)）
    t = torch.randn(dims_tuple, dtype=dtype)
    
    # Step 2: 用 as_strided 改 stride
    if strides is not None:
        try:
            t = torch.as_strided(t, size=dims_tuple, stride=strides)
        except RuntimeError:
            pass  # 静默回退到连续张量
    return t
```

**隐含假设**：`as_strided` 只需要 `numel` 个元素的底层存储。这在非连续张量上不成立。

### 1.2 问题场景

| 场景 | 当前行为 | 问题 |
|------|----------|------|
| **stride 超过 numel** | ❌ `RuntimeError: out of bounds`，静默回退到连续张量 | 测试用例与原始张量布局不一致 |
| **负 stride** | ❌ `RuntimeError: Negative strides are not supported`，静默回退 | 同上，且 PyTorch 原生不支持 |
| **stride=0（广播）** | ✅ 功能正确 | 内存浪费：分配 `numel` 个元素，实际需要 1 个 |
| **重叠内存** | ✅ 功能正确 | 无问题 |

### 1.3 静默回退的危害

```python
except RuntimeError:
    pass  # 失败时不报错，返回连续张量
```

- 测试用例与原始算子的输入布局不一致
- 可能掩盖 stride 相关的 bug（如某些算子对非连续输入处理有误）
- 用户无法感知到张量重建失败

---

## 2. 设计方案

### 2.1 核心思路

**预计算存储大小**：在创建张量前，先根据 `size` 和 `stride` 计算 `as_strided` 所需的底层存储大小和 offset，然后创建**恰好足够大**的连续张量，最后通过 `as_strided` 创建视图。

```
当前: torch.randn(numel) → as_strided (可能失败) → 连续张量
新设计: 计算 storage_size → torch.randn(storage_size) → as_strided (必然成功)
```

### 2.2 存储大小计算算法

对于给定的 `size` 和 `stride`，计算底层存储的公式：

```python
def _compute_storage_requirements(size, stride):
    """
    计算 as_strided 所需的底层存储大小和 offset。
    
    原理：
    - 每个维度的索引范围是 [0, size[i]-1]
    - 该维度的 offset 范围是 [0, (size[i]-1) * stride[i]]（正 stride）
      或 [(size[i]-1) * stride[i], 0]（负 stride）
    - 总 offset = 各维度 offset 之和
    - 最小总 offset 可能为负，需要 storage_offset 将其平移到非负
    - 存储大小 = 最大总 offset - 最小总 offset + 1
    
    示例 1: size=(2,3), stride=(100,1)
      dim0: offset 范围 [0, 100]
      dim1: offset 范围 [0, 2]
      总范围: [0, 102]
      storage_size = 102 - 0 + 1 = 103
      storage_offset = 0
    
    示例 2: size=(3,), stride=(-1,)
      dim0: offset 范围 [-2, 0]
      总范围: [-2, 0]
      storage_size = 0 - (-2) + 1 = 3
      storage_offset = 2  # 使索引 0 对应底层存储位置 2
    """
    dim_offsets = []
    for s, st in zip(size, stride):
        if st >= 0:
            start, end = 0, (s - 1) * st
        else:
            start, end = (s - 1) * st, 0
        dim_offsets.append((start, end))
    
    total_start = sum(start for start, _ in dim_offsets)
    total_end = sum(end for _, end in dim_offsets)
    storage_size = total_end - total_start + 1
    storage_offset = -total_start
    
    return storage_size, storage_offset
```

---

## 3. 详细设计

### 3.1 正 stride 处理（标准路径）

```python
def _build_tensor(self, dims, strides, dtype, op_name=""):
    # 解析 dims
    if isinstance(dims, list) and len(dims) > 0 and isinstance(dims[0], list):
        dims = dims[0]
    size_tuple = tuple(dims)
    
    # 无 stride 信息：创建连续张量（正常路径）
    if strides is None or len(strides) != len(size_tuple):
        return self._create_data_tensor(size_tuple, dtype, op_name)
    
    if isinstance(strides, list) and len(strides) > 0 and isinstance(strides[0], list):
        strides = strides[0]
    stride_tuple = tuple(strides)
    
    # 计算存储需求
    storage_size, storage_offset = self._compute_storage_requirements(
        size_tuple, stride_tuple
    )
    
    # 创建足够大的连续张量
    base_tensor = self._create_data_tensor(storage_size, dtype, op_name)
    
    # 创建视图
    return torch.as_strided(
        base_tensor,
        size=size_tuple,
        stride=stride_tuple,
        storage_offset=storage_offset
    )
```

### 3.2 负 stride 处理

PyTorch 的 `as_strided` 不支持负 stride。采用**正 stride + flip 模拟**：

```python
def _build_tensor(self, dims, strides, dtype, op_name=""):
    ...
    
    has_negative = any(st < 0 for st in stride_tuple)
    
    if not has_negative:
        # 标准路径（正 stride）
        return torch.as_strided(
            base_tensor, size_tuple, stride_tuple, storage_offset
        )
    else:
        # 负 stride 模拟路径
        abs_strides = tuple(abs(st) for st in stride_tuple)
        t = torch.as_strided(
            base_tensor, size_tuple, abs_strides, storage_offset
        )
        
        # 对负 stride 维度 flip，使数据遍历顺序与负 stride 一致
        flip_dims = [i for i, st in enumerate(stride_tuple) if st < 0]
        t = torch.flip(t, dims=flip_dims)
        
        return t
```

**重要说明**：
- `torch.flip` 创建的是**数据内容等价但内存布局不同**的新张量
- 对于功能正确性测试：**足够**（数据内容一致）
- 对于 in-place 操作测试：**可能不准确**（内存布局不同，某些算子行为依赖内存布局）
- 这是 PyTorch API 限制下的最佳近似方案

### 3.3 数据初始化逻辑提取

将现有的数据初始化逻辑提取为独立方法，供连续张量和底层张量共用：

```python
def _create_data_tensor(self, size, dtype: torch.dtype, op_name: str = "") -> torch.Tensor:
    """根据算子特性创建带适当初始值的张量"""
    if isinstance(size, int):
        size = (size,)
    
    base_name = op_name.replace("aten::", "").replace("aten::_", "")
    
    if self._is_floating(dtype):
        if base_name in ("rsqrt", "sqrt", "log", "log1p", "reciprocal"):
            t = torch.rand(size, dtype=dtype) + 0.1
        elif base_name in ("asin", "acos", "atan"):
            t = torch.rand(size, dtype=dtype) * 2 - 1
        elif base_name in ("acos",):
            t = torch.rand(size, dtype=dtype) * 1.8 - 0.9
        elif base_name in ("atanh",):
            t = torch.rand(size, dtype=dtype) * 1.8 - 0.9
        elif base_name in ("div", "true_divide", "floor_divide"):
            t = torch.randn(size, dtype=dtype)
            t = torch.where(t == 0, torch.ones_like(t), t)
        elif base_name in ("pow",):
            t = torch.randn(size, dtype=dtype).abs() + 0.1
        elif base_name in ("softmax",):
            t = torch.randn(size, dtype=dtype) * 20 - 10
        else:
            t = torch.randn(size, dtype=dtype) * 2 - 1
    elif dtype == torch.bool:
        t = torch.randint(low=0, high=2, size=size, dtype=dtype)
    else:
        t = torch.randint(low=0, high=10, size=size, dtype=dtype)
    
    return t
```

### 3.4 错误处理策略（不静默失败）

```python
def _build_tensor(self, dims, strides, dtype, op_name=""):
    ...
    
    if strides is None or len(strides) != len(size_tuple):
        # 无 stride 信息：这是正常路径，不警告
        return self._create_data_tensor(size_tuple, dtype, op_name)
    
    try:
        # 预计算 + 创建
        storage_size, storage_offset = self._compute_storage_requirements(...)
        base_tensor = self._create_data_tensor(storage_size, dtype, op_name)
        
        has_negative = any(st < 0 for st in stride_tuple)
        
        if not has_negative:
            return torch.as_strided(base_tensor, size_tuple, stride_tuple, storage_offset)
        else:
            # 负 stride 模拟
            abs_strides = tuple(abs(st) for st in stride_tuple)
            t = torch.as_strided(base_tensor, size_tuple, abs_strides, storage_offset)
            flip_dims = [i for i, st in enumerate(stride_tuple) if st < 0]
            return torch.flip(t, dims=flip_dims)
            
    except Exception as e:
        # 明确记录失败信息，不静默回退
        import warnings
        warnings.warn(
            f"TensorBuilder 无法为算子 '{op_name}' 创建非连续张量: {e}. "
            f"size={size_tuple}, stride={stride_tuple}. "
            f"回退到连续张量。",
            RuntimeWarning,
            stacklevel=2
        )
        return self._create_data_tensor(size_tuple, dtype, op_name)
```

**关键改进：**
1. 区分"无 stride 信息"（正常路径，不警告）和"有 stride 但创建失败"（异常情况，明确警告）
2. 使用 `warnings.warn()` 而非 `print()`，便于用户通过 warning 过滤器控制
3. 警告信息包含算子名、size、stride，便于调试

---

## 4. 边界情况处理

### 4.1 stride 超过 numel

**场景**：`size=(2,3), stride=(100,1)`

**当前**：`as_strided` 报 `out of bounds`，静默回退到连续张量

**新实现**：
- `storage_size = 1*100 + 2*1 + 1 = 103`
- `storage_offset = 0`
- 创建 103 个元素的底层张量，成功创建非连续视图

### 4.2 stride=0（广播张量）

**场景**：`size=(1000000,), stride=(0,)`

**当前**：创建 1,000,000 个元素的连续张量，再 `as_strided` 改 stride。功能正确但内存浪费 1,000,000 倍。

**新实现**：
- `storage_size = (1000000-1)*0 + 1 = 1`
- `storage_offset = 0`
- 创建 1 个元素的底层张量，成功创建广播视图
- 内存最优

### 4.3 负 stride

**场景**：`size=(3,), stride=(-1,)`

**当前**：`as_strided` 直接不支持，静默回退到连续张量

**新实现**：
- `storage_size = 3`, `storage_offset = 2`
- 创建 3 个元素的底层张量
- 用 `abs(stride)=(1,)` 创建视图
- 对维度 0 `torch.flip`
- 结果：数据内容等价，内存布局不同

### 4.4 混合正负 stride

**场景**：`size=(2,3), stride=(3,-1)`

**当前**：`as_strided` 不支持负值，静默回退

**新实现**：
- `storage_size = 5`, `storage_offset = 2`
- 用 `abs(stride)=(3,1)` 创建视图
- 对维度 1 `torch.flip`
- 结果：功能等价

### 4.5 重叠内存（非广播）

**场景**：`size=(2,2), stride=(1,1)`

**当前**：`as_strided` 成功，底层 3 个元素支撑 4 个逻辑位置

**新实现**：
- `storage_size = 3`, `storage_offset = 0`
- 行为与当前一致

---

## 5. 影响范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `op_testgen/builder/tensor_builder.py` | **重构** | `_build_tensor()` 改为预计算模式；新增 `_create_data_tensor()` 和 `_compute_storage_requirements()` |
| `tests/test_tensor_builder.py` | **新增** | 增加边界情况测试用例（大 stride、广播、负 stride、混合 stride） |
| `op_testgen/cli.py` | 无变更 | 接口不变 |
| `op_testgen/correctness/test_runner.py` | 无变更 | 接口不变 |
| `op_testgen/perf/benchmark.py` | 无变更 | 接口不变 |

---

## 6. 验收标准

- [ ] stride 超过 numel 时成功创建非连续张量（如 `size=(2,3), stride=(100,1)`）
- [ ] stride=0 广播张量内存最优（底层存储大小 = 1）
- [ ] 负 stride 张量通过 `flip` 模拟创建，功能等价
- [ ] 混合正负 stride 正确处理
- [ ] 创建失败时不静默回退，发出 `RuntimeWarning`
- [ ] 无 stride 信息时不发出警告（正常路径）
- [ ] 现有测试用例全部通过（`pytest tests/test_tensor_builder.py -v`）
- [ ] 新增边界测试用例通过
- [ ] 端到端测试正常（`op_testgen test profiler_trace.json --only-correctness --backend cpu`）

---

*文档版本: v1.0*
*作者: Sisyphus*
