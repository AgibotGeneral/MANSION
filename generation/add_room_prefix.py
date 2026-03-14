#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 JSON 文件中的所有 roomid 添加前缀 F*_
"""

import json
import re
import sys
from typing import Set, Dict, Any


def find_all_roomids(data: Dict[str, Any]) -> Set[str]:
    """找到所有出现的 roomid"""
    roomids = set()
    
    def extract_from_id_string(s: str):
        """从 id 字符串中提取 roomid"""
        if not isinstance(s, str):
            return
        
        # 从管道分隔的字符串中提取，如 "door|0|exterior|living_room" 或 "wall|living_room|outer|0"
        if '|' in s:
            parts = s.split('|')
            # 排除第一个部分（通常是类型标识符，如 door, wall, light, window）
            type_identifiers = ['exterior', 'interior', 'outer', 'door', 'wall', 'light', 'window']
            for i, part in enumerate(parts):
                part = part.strip()
                # 排除纯数字、特殊值和类型标识符
                if part and not part.isdigit() and part not in type_identifiers:
                    # 检查是否是 roomid（允许大写字母和&符号，如 Entrance_&_Checkout_Area）
                    # 排除纯数字和类型标识符
                    if re.match(r'^[a-zA-Z][a-zA-Z0-9_&]*$', part):
                        roomids.add(part)
        
        # 从括号中提取，如 "console_table-0 (Entrance_&_Checkout_Area)"
        if '(' in s:
            match = re.search(r'\(([a-zA-Z0-9_&]+)\)', s)
            if match:
                roomid_candidate = match.group(1)
                # 只接受看起来像 roomid 的值（字母开头）
                if re.match(r'^[a-zA-Z][a-zA-Z0-9_&]*$', roomid_candidate):
                    roomids.add(roomid_candidate)
    
    def traverse(obj, parent_key=None):
        """递归遍历 JSON 对象，只从特定字段提取 roomid"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                # 跳过 assetId 字段
                if key == 'assetId':
                    continue
                
                # 只从这些字段中提取 roomid
                if key in ['id', 'roomId', 'room0', 'room1', 'wall0', 'wall1'] and isinstance(value, str):
                    if key in ['roomId', 'room0', 'room1']:
                        # 直接是 roomid
                        if value and value != 'exterior':
                            # 只接受看起来像 roomid 的值（放宽限制）
                            if re.match(r'^[a-zA-Z][a-zA-Z0-9_&]*$', value):
                                roomids.add(value)
                    elif key == 'id':
                        # id 字段可能包含 roomid，需要解析
                        if '|' in value or '(' in value:
                            # 管道分隔或包含括号的 id，需要解析
                            extract_from_id_string(value)
                        elif parent_key == 'rooms':
                            # rooms 数组中的 id 字段，直接是 roomid
                            if value and value != 'exterior':
                                # 只接受看起来像 roomid 的值（放宽限制）
                                if re.match(r'^[a-zA-Z][a-zA-Z0-9_&]*$', value):
                                    roomids.add(value)
                
                traverse(value, key)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item, parent_key)
        # 不再处理普通字符串，避免误识别
    
    traverse(data, None)
    
    return roomids


def add_prefix_to_string(s: str, roomid_mapping: Dict[str, str]) -> str:
    """在字符串中替换所有 roomid"""
    if not isinstance(s, str):
        return s
    
    result = s
    # 按长度降序排序，避免短字符串先匹配导致长字符串无法匹配
    sorted_roomids = sorted(roomid_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    for old_roomid, new_roomid in sorted_roomids:
        # 如果字符串中不包含旧的 roomid，跳过
        if old_roomid not in result:
            continue
        
        # 替换管道分隔中的 roomid，如 "door|0|exterior|living_room" -> "door|0|exterior|F1_living_room"
        # 使用单词边界确保精确匹配
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(old_roomid) + r'(?![a-zA-Z0-9_])'
        result = re.sub(pattern, new_roomid, result)
    
    return result


_DEBUG_KEYS_TO_REMOVE = {
    'debug_object_selection_prompt',
    'debug_object_constraint_prompt',
    'debug_parsed_constraints',
    'raw_object_selection_llm',
    'raw_object_constraint_llm',
    'initial_selection_plan',
    'object_selection_plan',
    'original_floorplan',
    'portable_floorplan_edges',
    'artifacts_dir',
    'debug_artifacts_dir',
    'debug_dir',
}


def strip_debug_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """移除中间调试数据，只保留 AI2-THOR 运行时需要的字段"""
    return {k: v for k, v in data.items() if k not in _DEBUG_KEYS_TO_REMOVE}


def add_prefix_to_data(data: Dict[str, Any], prefix: str, clean: bool = True) -> Dict[str, Any]:
    """给所有 roomid 添加前缀，可选清理 debug 字段"""
    if clean:
        data = strip_debug_keys(data)

    roomids = find_all_roomids(data)
    roomid_mapping = {rid: f"{prefix}{rid}" for rid in roomids}

    def traverse_and_replace(obj):
        """递归遍历并替换"""
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if key == 'assetId':
                    new_obj[key] = value
                elif key in ['id', 'roomId', 'room0', 'room1', 'wall0', 'wall1'] and isinstance(value, str):
                    new_obj[key] = add_prefix_to_string(value, roomid_mapping)
                else:
                    new_obj[key] = traverse_and_replace(value)
            return new_obj
        elif isinstance(obj, list):
            return [traverse_and_replace(item) for item in obj]
        elif isinstance(obj, str):
            return obj
        else:
            return obj

    return traverse_and_replace(data)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="给场景 JSON 的 roomid 添加 FN_ 前缀")
    ap.add_argument("input_file", help="输入 JSON 文件")
    ap.add_argument("prefix", help="前缀，如 F1_")
    ap.add_argument("output_file", nargs="?", default=None, help="输出文件（默认 <input>_prefixed.json）")
    ap.add_argument("--no-clean", action="store_true", help="保留 debug/中间数据字段")
    args = ap.parse_args()

    prefix = args.prefix if args.prefix.endswith('_') else args.prefix + '_'
    output_file = args.output_file or args.input_file.replace('.json', '_prefixed.json')

    print(f"读取文件: {args.input_file}")
    print(f"使用前缀: {prefix}")
    print(f"输出文件: {output_file}")
    print(f"清理 debug 字段: {not args.no_clean}")

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified_data = add_prefix_to_data(data, prefix, clean=not args.no_clean)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)

    print(f"完成！已保存到: {output_file}")


if __name__ == '__main__':
    main()

