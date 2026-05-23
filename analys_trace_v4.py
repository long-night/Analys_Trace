#!/usr/bin/env python3
"""
Chrome Trace 算子分析脚本
解析 PyTorch profiler 导出的 chrome trace JSON 文件

输出格式:
- 如果安装了 openpyxl: 输出单个 Excel 文件（两个工作表）
- 如果未安装 openpyxl: 输出两个 CSV 文件

时间单位: 毫秒(ms)
"""

import json
import csv
from collections import defaultdict
from typing import Dict, List
import argparse
import sys

# 检测是否安装了 openpyxl
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


class ShapeStats:
    """Shape 统计信息"""
    def __init__(self, shape, strides=None):
        self.shape = shape
        self.strides = strides
        self.call_count = 0
        self.total_duration = 0.0  # 微秒
        self.durations = []
    
    def add_call(self, duration: float):
        """添加一次调用记录"""
        self.call_count += 1
        self.total_duration += duration
        self.durations.append(duration)
    
    def get_avg_duration(self):
        """获取平均执行时间（毫秒）"""
        avg_us = self.total_duration / self.call_count if self.call_count > 0 else 0.0
        return avg_us / 1000.0
    
    def get_min_duration(self):
        """获取最小执行时间（毫秒）"""
        return min(self.durations) / 1000.0 if self.durations else 0.0
    
    def get_max_duration(self):
        """获取最大执行时间（毫秒）"""
        return max(self.durations) / 1000.0 if self.durations else 0.0
    
    def get_total_duration_ms(self):
        """获取总执行时间（毫秒）"""
        return self.total_duration / 1000.0


class OperatorInfo:
    """算子信息类"""
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.total_duration = 0.0  # 微秒
        self.shape_stats = {}  # shape_str -> ShapeStats
    
    def add_shape(self, shape, strides=None, duration: float = 0.0):
        """添加 shape 信息"""
        if shape:
            shape_tuple = tuple(tuple(s) if isinstance(s, list) else s for s in shape)
            strides_tuple = None
            if strides:
                strides_tuple = tuple(tuple(s) if isinstance(s, list) else s for s in strides)

            key = str(list(shape_tuple))
            if strides_tuple:
                key += " | " + str(list(strides_tuple))

            if key not in self.shape_stats:
                self.shape_stats[key] = ShapeStats(shape_tuple, strides_tuple)

            self.shape_stats[key].add_call(duration)
    
    def get_total_duration_ms(self):
        """获取总执行时间（毫秒）"""
        return self.total_duration / 1000.0
    
    def get_avg_duration_ms(self):
        """获取平均执行时间（毫秒）"""
        if self.call_count > 0:
            return self.total_duration / self.call_count / 1000.0
        return 0.0
    
    def get_all_shapes_str(self):
        """获取所有 shape 的字符串表示"""
        if not self.shape_stats:
            return 'N/A'
        shapes = set()
        for stats in self.shape_stats.values():
            shapes.add(str(list(stats.shape)))
        return " | ".join(sorted(shapes))

    def get_all_strides_str(self):
        """获取所有 strides 的字符串表示"""
        if not self.shape_stats:
            return 'N/A'
        strides = set()
        for stats in self.shape_stats.values():
            if stats.strides:
                strides.add(str(list(stats.strides)))
        if not strides:
            return 'N/A'
        return " | ".join(sorted(strides))


class ChromeTraceAnalyzer:
    """Chrome Trace 分析器"""
    
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.operators = {}  # op_name -> OperatorInfo
    
    def load_trace(self):
        """加载 trace 文件"""
        print(f"正在加载 trace 文件: {self.trace_file}")
        try:
            with open(self.trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
            print(f"成功加载，共 {len(trace_data.get('traceEvents', []))} 个事件")
            return trace_data
        except json.JSONDecodeError as e:
            print(f"错误：无法解析 JSON 文件: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"错误：文件不存在: {self.trace_file}")
            sys.exit(1)
    
    def is_cpu_operator(self, event: dict) -> bool:
        """判断是否为 CPU 算子事件"""
        name = event.get('name', '')
        cat = event.get('cat', '')
        
        return (name.startswith('aten::') or name.startswith('torch::')) and \
               ('cpu' in cat.lower() or cat == 'cpu_op' or 'kernel' not in cat.lower())
    
    def extract_shapes(self, event: dict) -> list:
        """提取算子的 input shapes"""
        args = event.get('args', {})

        for key in ['Input Dims', 'input_dims', 'Input type', 'input_type', 'Input Shapes']:
            if key in args:
                dims = args[key]
                if isinstance(dims, list):
                    return dims

        return None

    def extract_strides(self, event: dict) -> list:
        """提取算子的 input strides"""
        args = event.get('args', {})

        for key in ['Input Strides', 'input_strides']:
            if key in args:
                strides = args[key]
                if isinstance(strides, list):
                    return strides

        return None
    
    def analyze(self):
        """分析 trace 文件"""
        trace_data = self.load_trace()
        events = trace_data.get('traceEvents', [])
        
        print("正在分析算子性能数据...")
        
        # 记录每个线程的 pending 事件
        pending_events = defaultdict(dict)  # tid -> {op_name: (ts, shapes)}
        
        for event in events:
            if not self.is_cpu_operator(event):
                continue
            
            name = event.get('name', '')
            tid = event.get('tid', 0)
            ph = event.get('ph', '')
            ts = event.get('ts', 0)
            
            # 确保算子记录存在
            if name not in self.operators:
                self.operators[name] = OperatorInfo(name)
            
            # 处理 Begin 事件
            if ph == 'B':
                shapes = self.extract_shapes(event)
                strides = self.extract_strides(event)
                pending_events[tid][name] = (ts, shapes, strides)
            
            # 处理 End 事件
            elif ph == 'E':
                if name in pending_events[tid]:
                    start_ts, shapes, strides = pending_events[tid][name]
                    duration = ts - start_ts

                    if shapes:
                        self.operators[name].add_shape(shapes, strides, duration)
                    
                    self.operators[name].call_count += 1
                    self.operators[name].total_duration += duration
                    
                    del pending_events[tid][name]
            
            # 处理完整事件 (ph == 'X')
            elif ph == 'X':
                duration = event.get('dur', 0.0)
                shapes = self.extract_shapes(event)
                strides = self.extract_strides(event)

                if shapes:
                    self.operators[name].add_shape(shapes, strides, duration)
                
                self.operators[name].call_count += 1
                self.operators[name].total_duration += duration
        
        print(f"分析完成，共发现 {len(self.operators)} 个不同的 CPU 算子")
    
    def export_to_csv(self, operators_file: str, shapes_file: str):
        """导出到两个 CSV 文件"""
        print(f"\n正在导出到 CSV 文件...")
        
        # ===== CSV 1: 算子总表 =====
        print(f"  - 正在写入: {operators_file}")
        
        # 按总执行时间降序排序
        sorted_ops = sorted(self.operators.values(), 
                           key=lambda x: x.total_duration, 
                           reverse=True)
        
        with open(operators_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['算子名称', '输入Shapes', '输入Strides', '调用次数', '总执行时间(ms)', '平均执行时间(ms)']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for op in sorted_ops:
                writer.writerow({
                    '算子名称': op.name,
                    '输入Shapes': op.get_all_shapes_str(),
                    '输入Strides': op.get_all_strides_str(),
                    '调用次数': op.call_count,
                    '总执行时间(ms)': round(op.get_total_duration_ms(), 4),
                    '平均执行时间(ms)': round(op.get_avg_duration_ms(), 4)
                })
        
        print(f"    ✓ 已写入 {len(sorted_ops)} 个算子")
        
        # ===== CSV 2: Shape 统计表 =====
        print(f"  - 正在写入: {shapes_file}")
        
        # 收集所有 shape 统计信息
        all_shape_stats = []
        for op in self.operators.values():
            for key, stats in op.shape_stats.items():
                strides_str = str(list(stats.strides)) if stats.strides else 'N/A'
                all_shape_stats.append({
                    'op_name': op.name,
                    'shape': str(list(stats.shape)),
                    'strides': strides_str,
                    'call_count': stats.call_count,
                    'total_duration_ms': stats.get_total_duration_ms(),
                    'avg_duration_ms': stats.get_avg_duration(),
                    'min_duration_ms': stats.get_min_duration(),
                    'max_duration_ms': stats.get_max_duration()
                })

        all_shape_stats.sort(key=lambda x: (x['op_name'], -x['total_duration_ms']))

        with open(shapes_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['算子名称', '输入Shapes', '输入Strides', '调用次数',
                         '总执行时间(ms)', '平均执行时间(ms)',
                         '最小执行时间(ms)', '最大执行时间(ms)']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for stat in all_shape_stats:
                writer.writerow({
                    '算子名称': stat['op_name'],
                    '输入Shapes': stat['shape'],
                    '输入Strides': stat['strides'],
                    '调用次数': stat['call_count'],
                    '总执行时间(ms)': round(stat['total_duration_ms'], 4),
                    '平均执行时间(ms)': round(stat['avg_duration_ms'], 4),
                    '最小执行时间(ms)': round(stat['min_duration_ms'], 4),
                    '最大执行时间(ms)': round(stat['max_duration_ms'], 4)
                })
        
        print(f"    ✓ 已写入 {len(all_shape_stats)} 条 Shape 统计记录")
        print(f"\n✓ CSV 文件已保存:")
        print(f"  - {operators_file}")
        print(f"  - {shapes_file}")
    
    def export_to_excel(self, output_file: str):
        """导出到 Excel 文件"""
        if not OPENPYXL_AVAILABLE:
            print("错误：openpyxl 未安装，无法导出 Excel 文件")
            return False
        
        print(f"\n正在导出到 Excel 文件: {output_file}")
        
        # 创建工作簿
        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # 删除默认工作表
        
        # 创建两个工作表
        ws_operators = wb.create_sheet("算子总表")
        ws_shapes = wb.create_sheet("Shape统计表")
        
        # 设置样式
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        header_alignment = Alignment(horizontal="center", vertical="center")
        
        # ===== 工作表1: 算子总表 =====
        print("  - 正在写入算子总表...")
        
        # 表头
        operators_headers = ['算子名称', '输入Shapes', '输入Strides', '调用次数', '总执行时间(ms)', '平均执行时间(ms)']
        ws_operators.append(operators_headers)
        
        # 设置表头样式
        for col_num, header in enumerate(operators_headers, 1):
            cell = ws_operators.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment
        
        # 按总执行时间降序排序
        sorted_ops = sorted(self.operators.values(), 
                           key=lambda x: x.total_duration, 
                           reverse=True)
        
        # 写入数据
        for op in sorted_ops:
            ws_operators.append([
                op.name,
                op.get_all_shapes_str(),
                op.get_all_strides_str(),
                op.call_count,
                round(op.get_total_duration_ms(), 4),
                round(op.get_avg_duration_ms(), 4)
            ])
        
        # 自动调整列宽
        for col in ws_operators.columns:
            max_length = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = min(max_length + 2, 80)  # 最大宽度限制为80
            ws_operators.column_dimensions[col_letter].width = adjusted_width
        
        print(f"    ✓ 已写入 {len(sorted_ops)} 个算子")
        
        # ===== 工作表2: Shape 统计表 =====
        print("  - 正在写入 Shape 统计表...")
        
        # 表头
        shape_headers = ['算子名称', '输入Shapes', '输入Strides', '调用次数',
                        '总执行时间(ms)', '平均执行时间(ms)',
                        '最小执行时间(ms)', '最大执行时间(ms)']
        ws_shapes.append(shape_headers)
        
        # 设置表头样式
        for col_num, header in enumerate(shape_headers, 1):
            cell = ws_shapes.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment
        
        # 收集所有 shape 统计信息
        all_shape_stats = []
        for op in self.operators.values():
            for key, stats in op.shape_stats.items():
                strides_str = str(list(stats.strides)) if stats.strides else 'N/A'
                all_shape_stats.append({
                    'op_name': op.name,
                    'shape': str(list(stats.shape)),
                    'strides': strides_str,
                    'call_count': stats.call_count,
                    'total_duration_ms': stats.get_total_duration_ms(),
                    'avg_duration_ms': stats.get_avg_duration(),
                    'min_duration_ms': stats.get_min_duration(),
                    'max_duration_ms': stats.get_max_duration()
                })

        all_shape_stats.sort(key=lambda x: (x['op_name'], -x['total_duration_ms']))

        for stat in all_shape_stats:
            ws_shapes.append([
                stat['op_name'],
                stat['shape'],
                stat['strides'],
                stat['call_count'],
                round(stat['total_duration_ms'], 4),
                round(stat['avg_duration_ms'], 4),
                round(stat['min_duration_ms'], 4),
                round(stat['max_duration_ms'], 4)
            ])
        
        # 自动调整列宽
        for col in ws_shapes.columns:
            max_length = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = min(max_length + 2, 80)
            ws_shapes.column_dimensions[col_letter].width = adjusted_width
        
        print(f"    ✓ 已写入 {len(all_shape_stats)} 条 Shape 统计记录")
        
        # 保存文件
        wb.save(output_file)
        print(f"\n✓ Excel 文件已保存: {output_file}")
        return True
    
    def print_summary(self):
        """打印摘要信息"""
        print("\n" + "=" * 60)
        print("分析摘要".center(60))
        print("=" * 60)
        
        # 最耗时的算子 TOP 10
        print("\n【最耗时的算子 TOP 10】")
        top_time_ops = sorted(self.operators.values(), 
                             key=lambda x: x.total_duration, 
                             reverse=True)[:10]
        
        total_time_all = sum(op.total_duration for op in self.operators.values())
        
        for i, op in enumerate(top_time_ops, 1):
            percentage = (op.total_duration / total_time_all * 100) if total_time_all > 0 else 0
            print(f"  {i:2d}. {op.name}")
            print(f"      总时间: {op.get_total_duration_ms():.2f} ms ({percentage:.1f}%), "
                  f"调用: {op.call_count} 次, "
                  f"平均: {op.get_avg_duration_ms():.4f} ms")
        
        # 调用最频繁的算子 TOP 5
        print("\n【调用最频繁的算子 TOP 5】")
        top_call_ops = sorted(self.operators.values(), 
                             key=lambda x: x.call_count, 
                             reverse=True)[:5]
        for i, op in enumerate(top_call_ops, 1):
            print(f"  {i}. {op.name}: {op.call_count} 次, "
                  f"总时间: {op.get_total_duration_ms():.2f} ms")
        
        # Shape 多样性
        print("\n【Shape 多样性最高的算子 TOP 5】")
        top_diversity = sorted(self.operators.values(), 
                              key=lambda x: len(x.shape_stats), 
                              reverse=True)[:5]
        for i, op in enumerate(top_diversity, 1):
            print(f"  {i}. {op.name}: {len(op.shape_stats)} 种不同的 shape")
        
        # 总体统计
        print(f"\n【总体统计】")
        print(f"  总算子数: {len(self.operators)}")
        print(f"  总执行时间: {total_time_all / 1000.0:.2f} ms")
        print(f"  总调用次数: {sum(op.call_count for op in self.operators.values())}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='分析 PyTorch Chrome Trace 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyze_trace.py trace.json
  python analyze_trace.py trace.json -o report.xlsx
  python analyze_trace.py trace.json --csv -o operators.csv
  python analyze_trace.py trace.json --no-summary
  
输出格式:
  - 默认: 如果安装了 openpyxl，输出 Excel 文件（两个工作表）
          如果未安装 openpyxl，输出两个 CSV 文件
  - 使用 --csv: 强制输出 CSV 文件
  - 使用 --excel: 强制输出 Excel 文件（需要 openpyxl）
        """
    )
    
    parser.add_argument('trace_file', help='Chrome trace JSON 文件路径')
    parser.add_argument('-o', '--output', 
                       help='输出文件路径（默认根据格式自动命名）')
    parser.add_argument('--csv', action='store_true',
                       help='强制输出为 CSV 格式（两个文件）')
    parser.add_argument('--excel', action='store_true',
                       help='强制输出为 Excel 格式（需要 openpyxl）')
    parser.add_argument('--no-summary', action='store_true',
                       help='不显示摘要信息')
    
    args = parser.parse_args()
    
    # 检查 openpyxl 安装情况
    if not OPENPYXL_AVAILABLE:
        print("=" * 60)
        print("⚠️  提示: openpyxl 库未安装")
        print("=" * 60)
        print("如需导出为 Excel 格式（单个文件，两个工作表），请安装:")
        print("  pip install openpyxl")
        print("\n当前将导出为 CSV 格式（两个独立文件）")
        print("=" * 60)
        print()
    
    # 决定输出格式
    force_csv = args.csv
    force_excel = args.excel
    
    if force_excel and not OPENPYXL_AVAILABLE:
        print("错误：指定了 --excel 但 openpyxl 未安装")
        print("请运行: pip install openpyxl")
        sys.exit(1)
    
    # 默认行为：有 openpyxl 就用 Excel，没有就用 CSV
    use_excel = OPENPYXL_AVAILABLE and not force_csv
    
    # 如果用户强制指定了格式，以用户指定为准
    if force_excel:
        use_excel = True
    elif force_csv:
        use_excel = False
    
    # 创建分析器
    analyzer = ChromeTraceAnalyzer(args.trace_file)
    
    # 执行分析
    analyzer.analyze()
    
    # 导出文件
    if use_excel:
        output_file = args.output if args.output else 'operator_analysis.xlsx'
        if not output_file.endswith('.xlsx'):
            output_file += '.xlsx'
        analyzer.export_to_excel(output_file)
    else:
        # CSV 模式
        if args.output:
            # 用户指定了输出文件名，用作算子总表文件名
            base_name = args.output.replace('.csv', '')
            operators_file = f"{base_name}_operators.csv"
            shapes_file = f"{base_name}_shapes.csv"
        else:
            operators_file = 'cpu_operators.csv'
            shapes_file = 'shape_statistics.csv'
        
        analyzer.export_to_csv(operators_file, shapes_file)
    
    # 显示摘要
    if not args.no_summary:
        analyzer.print_summary()
    
    print("\n✓ 全部完成！")


if __name__ == '__main__':
    main()
