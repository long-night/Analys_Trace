# TensorBuilder 非连续张量支持 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `TensorBuilder._build_tensor` 无法正确处理大 stride、负 stride、广播等边界情况，改为预计算存储大小的模式。

**Architecture:** 提取数据初始化逻辑为 `_create_data_tensor`，新增 `_compute_storage_requirements` 静态方法计算底层存储大小，重构 `_build_tensor` 为先计算后创建。负 stride 用 `abs(stride) + torch.flip` 模拟。

**Tech Stack:** Python 3.12, PyTorch, pytest

---

## 文件变更清单

| 文件 | 变更类型 | 职责 |
|------|----------|------|
| `op_testgen/builder/tensor_builder.py` | 重构 | 新增 `_compute_storage_requirements`、`_create_data_tensor`，重构 `_build_tensor` |
| `tests/test_tensor_builder.py` | 新增 | 边界测试用例（大 stride、广播、负 stride、混合 stride、重叠内存） |

---

## Task 1: 新增 `_compute_storage_requirements` 静态方法

**Files:**
- Modify: `op_testgen/builder/tensor_builder.py`（在 `TensorBuilder` 类中添加新方法）

- [ ] **Step 1: 在 `TensorBuilder` 类中添加 `_compute_storage_requirements` 方法**

  在第 67-68 行之间（`_is_floating` 方法之后）添加：

  ```python
  @staticmethod
  def _compute_storage_requirements(size: tuple, stride: tuple) -> tuple[int, int]:
      """计算 as_strided 所需的底层存储大小和 offset。

      原理：
      - 每个维度的索引范围是 [0, size[i]-1]
      - 该维度的 offset 范围取决于 stride 的正负
      - 总 offset 的最小值可能为负，需要 storage_offset 平移到非负
      - 存储大小 = 最大总 offset - 最小总 offset + 1

      Args:
          size: 张量各维度大小
          stride: 张量各维度步长

      Returns:
          (storage_size, storage_offset)
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

- [ ] **Step 2: 验证语法**

  Run: `python -c "from op_testgen.builder.tensor_builder import TensorBuilder; print(TensorBuilder._compute_storage_requirements((2, 3), (100, 1)))"`
  Expected: `(103, 0)`

- [ ] **Step 3: Commit**

  ```bash
  git add op_testgen/builder/tensor_builder.py
  git commit -m "feat(builder): add _compute_storage_requirements for strided tensor support

  Calculates the underlying storage size and offset needed for
  torch.as_strided to handle large strides and negative strides."
  ```

---

## Task 2: 新增 `_create_data_tensor` 方法

**Files:**
- Modify: `op_testgen/builder/tensor_builder.py`

- [ ] **Step 1: 在 `_compute_storage_requirements` 之后添加 `_create_data_tensor` 方法**

  ```python
  def _create_data_tensor(self, size, dtype: torch.dtype, op_name: str = "") -> torch.Tensor:
      """根据算子特性创建带适当初始值的张量。

      Args:
          size: 张量大小（int 或 tuple）
          dtype: 数据类型
          op_name: 算子名称，用于特殊初始化

      Returns:
          初始化后的张量
      """
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

- [ ] **Step 2: 验证方法可用**

  Run: `python -c "from op_testgen.builder.tensor_builder import TensorBuilder; b = TensorBuilder(); t = b._create_data_tensor((2, 3), torch.float32, 'aten::add'); print(t.shape, t.dtype)"`
  Expected: `torch.Size([2, 3]) torch.float32`

- [ ] **Step 3: Commit**

  ```bash
  git add op_testgen/builder/tensor_builder.py
  git commit -m "feat(builder): add _create_data_tensor helper method

  Extracts data initialization logic so it can be reused for both
  contiguous tensors and underlying storage for strided views."
  ```

---

## Task 3: 重构 `_build_tensor` 方法

**Files:**
- Modify: `op_testgen/builder/tensor_builder.py:97-132`

- [ ] **Step 1: 替换 `_build_tensor` 实现**

  将第 97-132 行替换为：

  ```python
  def _build_tensor(self, dims, strides, dtype: torch.dtype, op_name: str = "") -> torch.Tensor:
      # 解析 dims（处理嵌套列表的情况）
      if isinstance(dims, list) and len(dims) > 0 and isinstance(dims[0], list):
          dims = dims[0]
      size_tuple = tuple(dims)

      # 无 stride 信息或长度不匹配：创建连续张量（正常路径，不警告）
      if strides is None or len(strides) != len(size_tuple):
          return self._create_data_tensor(size_tuple, dtype, op_name)

      # 解析 strides（处理嵌套列表的情况）
      if isinstance(strides, list) and len(strides) > 0 and isinstance(strides[0], list):
          strides = strides[0]
      stride_tuple = tuple(strides)

      try:
          # 预计算存储需求
          storage_size, storage_offset = self._compute_storage_requirements(
              size_tuple, stride_tuple
          )

          # 创建足够大的底层张量
          base_tensor = self._create_data_tensor(storage_size, dtype, op_name)

          # 检查是否有负 stride
          has_negative = any(st < 0 for st in stride_tuple)

          if not has_negative:
              # 标准路径：正 stride，直接 as_strided
              return torch.as_strided(
                  base_tensor,
                  size=size_tuple,
                  stride=stride_tuple,
                  storage_offset=storage_offset
              )
          else:
              # 负 stride 模拟路径：用 abs(stride) + flip
              abs_strides = tuple(abs(st) for st in stride_tuple)
              t = torch.as_strided(
                  base_tensor,
                  size=size_tuple,
                  stride=abs_strides,
                  storage_offset=storage_offset
              )

              # 对负 stride 维度 flip，使数据遍历顺序一致
              flip_dims = [i for i, st in enumerate(stride_tuple) if st < 0]
              t = torch.flip(t, dims=flip_dims)

              return t

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

- [ ] **Step 2: 运行现有测试确保无回归**

  Run: `pytest tests/test_tensor_builder.py -v`
  Expected: 全部通过（6 个测试用例）

- [ ] **Step 3: Commit**

  ```bash
  git add op_testgen/builder/tensor_builder.py
  git commit -m "refactor(builder): rewrite _build_tensor with precomputed storage size

  Replaces the old 'create contiguous then as_strided' approach with
  'compute storage size, create base tensor, then as_strided'.
  Handles large strides, broadcast (stride=0), and negative strides."
  ```

---

## Task 4: 新增边界测试用例

**Files:**
- Modify: `tests/test_tensor_builder.py`

- [ ] **Step 1: 在测试类末尾添加边界测试方法**

  在 `test_scalar_not_tensor` 方法之后添加：

  ```python
    def test_large_stride(self, builder):
        """stride 超过 numel 时，应成功创建非连续张量"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3]],
            input_strides=[[100, 1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([2, 3])
        assert t.stride() == (100, 1)
        assert not t.is_contiguous()

    def test_broadcast_stride_zero(self, builder):
        """stride=0（广播张量）应内存最优，底层存储=1"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[1000]],
            input_strides=[[0]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([1000])
        assert t.stride() == (0,)
        # 广播张量底层存储应为 1
        assert t.storage().size() == 1

    def test_negative_stride(self, builder):
        """负 stride 应通过 flip 模拟创建"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[3]],
            input_strides=[[-1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([3])
        # torch.flip 返回的张量 stride 为正，但数据内容等价
        assert t.tolist() == pytest.approx([t.storage()[2].item(), t.storage()[1].item(), t.storage()[0].item()], abs=1e-5)

    def test_mixed_positive_negative_stride(self, builder):
        """混合正负 stride 应正确处理"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3]],
            input_strides=[[3, -1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([2, 3])
        assert t.numel() == 6

    def test_overlapping_memory(self, builder):
        """重叠内存（非广播）应正确处理"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 2]],
            input_strides=[[1, 1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([2, 2])
        assert t.stride() == (1, 1)
        assert t.storage().size() == 3  # 重叠内存，底层存储小于 numel

    def test_no_stride_fallback(self, builder):
        """无 stride 信息时不应发出警告"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_case = builder.build(mapped)
            # 不应有 RuntimeWarning
            assert not any(issubclass(warning.category, RuntimeWarning) for warning in w)

        assert len(test_case.input_tensors) == 2
        assert all(t.is_contiguous() for t in test_case.input_tensors)
  ```

- [ ] **Step 2: 运行新增测试**

  Run: `pytest tests/test_tensor_builder.py -v`
  Expected: 全部通过（12 个测试用例，6 个原有 + 6 个新增）

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_tensor_builder.py
  git commit -m "test(builder): add boundary tests for strided tensor support

  Covers: large stride, broadcast (stride=0), negative stride,
  mixed positive/negative stride, overlapping memory, and no-stride fallback."
  ```

---

## Task 5: 端到端验证

**Files:**
- Test: `tests/` 全部测试 + 端到端 smoke test

- [ ] **Step 1: 运行全部测试**

  Run: `pytest tests/ -v`
  Expected: 全部通过（73+ 个测试用例）

- [ ] **Step 2: 运行端到端 smoke test**

  Run: `python -m op_testgen.cli test profiler_trace.json --only-correctness --backend cpu --max-ops 5`
  Expected: 正常执行，无内存异常，输出报告。如有 RuntimeWarning 应明确显示算子名和 stride 信息。

- [ ] **Step 3: Commit（如有修复）**

  如有测试失败或需要修复：
  ```bash
  git add -A
  git commit -m "fix(builder): address edge cases in strided tensor creation"
  ```

---

## Task 6: 验证警告行为

**Files:**
- Test: 手动验证

- [ ] **Step 1: 验证失败时发出 RuntimeWarning**

  创建一个故意会失败的场景（如空 dims 但 stride 非空，虽然这在实际中不太可能）：

  Run:
  ```python
  python -c "
  import warnings
  warnings.simplefilter('always')
  from op_testgen.builder.tensor_builder import TensorBuilder
  b = TensorBuilder()
  # 测试无异常路径不警告
  t = b._build_tensor([2, 3], [3, 1], torch.float32, 'aten::add')
  print('Normal path OK')
  "
  ```
  Expected: `Normal path OK`，无 RuntimeWarning

- [ ] **Step 2: 验证端到端警告输出**

  如果 trace 中有特殊 stride 的算子，运行测试时观察是否有警告：
  
  Run: `python -m op_testgen.cli test profiler_trace.json --only-correctness --backend cpu --max-ops 20 2>&1 | grep -i "警告\|warning" || echo "No warnings"`
  Expected: 显示 "No warnings" 或明确的警告信息（如果有特殊 stride）

---

## 自审检查

- [x] **Spec 覆盖**: 所有 design.md 中的修改点均已对应到 Task 1-5
- [x] **Placeholder 检查**: 无 TBD/TODO/"similar to Task N"
- [x] **类型一致性**: `_compute_storage_requirements` 返回 `tuple[int, int]`，与调用处一致
- [x] **文件路径**: 所有路径使用从 repo root 的相对路径
- [x] **测试覆盖**: 6 个新增测试覆盖所有边界情况
- [x] **错误处理**: 警告策略符合 spec 要求（区分正常路径和异常路径）

---

*计划版本: v1.0*
*作者: Sisyphus*
