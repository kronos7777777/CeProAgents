import win32com.client
import os
import time
from typing import Dict, Union, List, Tuple, Any
import gc


def run_aspen_simulation(
        aspen_file_path: str,
        input_variables: Dict[str, Union[float, int, str]],
        output_variables: List[str],
        visible: bool = False,
        max_wait_time: int = 300
) -> Dict[str, Union[float, int, str, None]]:
    aspen_app = None
    try:
        aspen_app = win32com.client.Dispatch("Apwn.Document")
        aspen_app.InitFromArchive2(os.path.abspath(aspen_file_path))
        aspen_app.Visible = visible
        tree = aspen_app.Tree

        for path, value in input_variables.items():
            node = tree.FindNode(path)
            if node:
                node.Value = value
            else:
                print(f"[Warning] 节点未找到（设置失败）: {path}")

        aspen_app.Engine.Run2()
        start = time.time()
        while time.time() - start < max_wait_time:
            if aspen_app.Engine.IsRunning == 0:
                break
            time.sleep(1)
        else:
            print(f"[Error] 模拟超时（>{max_wait_time}s）")
            return {path: None for path in output_variables}

        path = r"\Data\Blocks"
        node = tree.FindNode(path)
        results = {}
        for path in output_variables:
            node = tree.FindNode(path)
            if node:
                val = node.Value
                unit = node.UnitString or ""
                results[path] = (val, unit)
            if node is None:
                print(f"[Warning] 节点未找到（读取失败）: {path}")
                results[path] = (None, "")
        return results

    except Exception as e:
        print(f"[Error] 模拟执行异常: {e}")
        return {path: None for path in output_variables}

    finally:
        if aspen_app:
            try:
                aspen_app.Close()
            except Exception:
                pass


def extract_aspen_flowsheet_connections(aspen_file_path: str,
                                        visible: bool = False,
                                        ) -> Dict[str, Dict[str, List[str]]]:
    aspen_app = None
    try:
        # 启动 Aspen
        aspen_app = win32com.client.Dispatch("Apwn.Document")
        aspen_app.InitFromArchive2(os.path.abspath(aspen_file_path))
        aspen_app.Visible = visible
        tree = aspen_app.Tree
        blocks_node = tree.FindNode(r"\Data\Blocks")

        if not blocks_node:
            raise RuntimeError("未找到 \\Data\\Blocks 节点")
        connections_graph = {}

        for block in blocks_node.Elements:
            block_id = block.Name
            block_type = block.AttributeValue(6)
            conn_path = f"\\Data\\Blocks\\{block_id}\\Connections"
            conn_node = tree.FindNode(conn_path)
            if not conn_node:
                continue  
            in_streams = []
            out_streams = []

            for conn in conn_node.Elements:
                stream_id = conn.Name
                conn_type = conn.Value  
                if "IN" in conn_type:
                    in_streams.append(stream_id)
                elif "OUT" in conn_type:
                    out_streams.append(stream_id)

            connections_graph[(block_id, block_type)] = {
                "IN": in_streams,
                "OUT": out_streams
            }

        graph = convert_to_graph(connections_graph)
        return graph
    except Exception as e:
        print(f"[Error] 模拟执行异常: {e}")
        return {}
    finally:

        if aspen_app:
            try:
                aspen_app.Close()
            except Exception:
                pass


def convert_to_graph(connection_graph):
    equipment_nodes = []
    equip_id_set = set()
    for (eid, etype) in connection_graph.keys():
        equipment_nodes.append({
            "identifier": eid,
            "type": etype
        })
        equip_id_set.add(eid)

    all_streams = set()
    for io in connection_graph.values():
        all_streams.update(io['IN'])
        all_streams.update(io['OUT'])

    stream_to_sources = {}   
    stream_to_targets = {}   

    for (eid, _), io in connection_graph.items():
        for out_s in io['OUT']:
            stream_to_sources[out_s] = eid
        for in_s in io['IN']:
            stream_to_targets[in_s] = eid

    feed_streams = [s for s in all_streams if s not in stream_to_sources]
    product_streams = [s for s in all_streams if s not in stream_to_targets]

    virtual_nodes = []
    for s in feed_streams:
        virtual_nodes.append({
            "identifier": f"FEED_{s}",
            "type": "NA"
        })
    for s in product_streams:
        virtual_nodes.append({
            "identifier": f"PRODUCT_{s}",
            "type": "NA"
        })

    connections = []

    for (eid, _), io in connection_graph.items():
        for in_s in io['IN']:
            if in_s in stream_to_sources:
                src = stream_to_sources[in_s]
            else:
                src = f"FEED_{in_s}"
            connections.append({
                "source": src,
                "target": eid
            })

    for out_s in product_streams:
        if out_s in stream_to_sources:
            src = stream_to_sources[out_s]
            connections.append({
                "source": src,
                "target": f"PRODUCT_{out_s}"
            })

    all_nodes = equipment_nodes + virtual_nodes

    return {
        "equipments": all_nodes,
        "connections": connections
    }


def parse_structured_inputs(config: Dict[Union[str, Tuple[str, str]], Any]) -> Tuple[Dict[str, Union[float, int, str]], List[str]]:
    input_dict = {}
    output_paths = []

    for key, value in config.items():
        if isinstance(key, tuple) and len(key) == 2:
            stream_id, component = key
            if isinstance(value, (int, float, str)) and not isinstance(value, list):
                path = f"\\Data\\Streams\\{stream_id}\\Input\\FLOW\\MIXED\\{component}"
                input_dict[path] = value
            elif isinstance(value, list):
                for var_name in value:
                    if not isinstance(var_name, str) or not var_name.isalpha():
                        raise ValueError(f"流股查询变量名必须为字母字符串：{var_name}")
                    path = f"\\Data\\Streams\\{stream_id}\\Output\\{var_name}\\MIXED\\{component}"
                    output_paths.append(path)
            else:
                raise ValueError(f"流股操作值格式错误：{value}")
        elif isinstance(key, str):
            if isinstance(value, str):
                base_path = f"\\Data\\Blocks\\{key}\\Output\\{value}"
                if value == "DUTY":
                    output_paths.append(f"{base_path.replace('DUTY', 'COND_DUTY')}")
                    output_paths.append(f"{base_path.replace('DUTY', 'REB_DUTY')}")
                else:
                    output_paths.append(base_path)

            elif isinstance(value, list):
                for entry in value:
                    if not isinstance(entry, (tuple, list)):
                        raise ValueError(f"设备参数条目必须是元组或列表：{entry}")
                    if len(entry) == 2:
                        param, val = entry
                        path = f"\\Data\\Blocks\\{key}\\Input\\{param}"
                        input_dict[path] = val
                    elif len(entry) == 3:
                        param, stream_id, val = entry
                        path = f"\\Data\\Blocks\\{key}\\Input\\{param}\\{stream_id}"
                        input_dict[path] = val
                    else:
                        raise ValueError(f"设备参数元组长度必须为2或3：{entry}")
            else:
                raise ValueError(f"设备 {key} 的值类型不支持：{type(value)}")
    return input_dict, output_paths


def group_output_results(
        output_paths: List[str],
        path_values: Dict[str, Tuple[Union[float, int, str, None], str]]
) -> Dict[Union[Tuple[str, str], str], Any]:
    result: Dict[Union[Tuple[str, str], str], Any] = {}

    for path in output_paths:
        if path not in path_values:
            raise KeyError(f"路径 {path} 在 path_values 中缺失")
        value, unit = path_values[path]
        if value is None:
            value = 0.0  

        parts = [p for p in path.split('\\') if p]
        if len(parts) >= 5 and parts[0] == "Data":
            if parts[1] == "Streams":
                # \Data\Streams\S31\Output\MOLEFLOW\MIXED\H2O
                stream_id = parts[2]
                prop = parts[4]
                component = parts[-1]
                key = (stream_id, prop)
                if key not in result:
                    result[key] = {}
                result[key][component] = (float(value), unit)
            elif parts[1] == "Blocks":
                # \Data\Blocks\R0401\Output\QCALC
                block_id = parts[2]
                var_name = parts[4]
                if block_id not in result:
                    result[block_id] = {}
                result[block_id][var_name] = (float(value), unit)
            else:
                raise ValueError(f"不支持的输出路径类型: {path}")
        else:
            raise ValueError(f"无效路径格式: {path}")
    return result


def run_aspen_with_structured_io(
    aspen_file_path: str,
    input_config: Dict[Union[str, Tuple[str, str]], Any],
    output_config: Dict[Union[str, Tuple[str, str]], List[str]],
    visible: bool = False,
    max_wait_time: int = 600
) -> Dict[Union[Tuple[str, str], str], Any]:

    full_config = input_config.copy()
    for key, val in output_config.items():
        if isinstance(key, str):
            if isinstance(val, str):
                full_config[key] = val
            else:
                raise ValueError(f"设备输出请求的值必须是字符串（如 'QCALC'），得到: {key}: {val}")
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(val, list) and all(isinstance(v, str) for v in val):
                full_config[key] = val
            else:
                raise ValueError(f"流股输出请求的值必须是字符串列表，得到: {key}: {val}")
        else:
            raise ValueError(f"output_config 的键必须是字符串（设备ID）或二元组（流股ID, 组分），得到: {key}")

    input_dict, output_paths = parse_structured_inputs(full_config)

    aspen_result = run_aspen_simulation(
        aspen_file_path=os.path.abspath(aspen_file_path),
        input_variables=input_dict,
        output_variables=output_paths,
        visible=visible,
        max_wait_time=max_wait_time
    )

    for path in output_paths:
        if path not in aspen_result:
            aspen_result[path] = None

    structured_result = group_output_results(output_paths, aspen_result)

    return structured_result


def calculate_conversion_selectivity_yield(
        feed_data: dict,
        product_data: dict,
        feed_stream_id: str,
        product_stream_id: str,
        reactant: str,
        product: str,
        stoich_reactant: float,
        stoich_product: float
) -> tuple[float, float, float]:

    feed_key = (feed_stream_id, 'MOLEFLOW')
    prod_key = (product_stream_id, 'MOLEFLOW')

    if feed_key not in feed_data:
        raise KeyError(f"未找到进口流股 {feed_stream_id} 的 MOLEFLOW 数据")
    if prod_key not in product_data:
        raise KeyError(f"未找到出口流股 {product_stream_id} 的 MOLEFLOW 数据")

    feed_comp = feed_data[feed_key]
    prod_comp = product_data[prod_key]

    F_in_R = feed_comp.get(reactant, 0.0)
    F_out_R = prod_comp.get(reactant, 0.0)
    F_in_P = feed_comp.get(product, 0.0)
    F_out_P = prod_comp.get(product, 0.0)

    if F_in_R <= 0:
        raise ValueError(f"反应物 {reactant} 在进口流股中流量为 {F_in_R}，无法计算")

    conversion = 1.0 - (F_out_R / F_in_R)

    reacted_R = F_in_R - F_out_R

    produced_P = F_out_P - F_in_P

    if reacted_R <= 0:
        selectivity = 0.0 if produced_P == 0 else float('inf')
    else:
        selectivity = (produced_P / stoich_product) / (reacted_R / stoich_reactant)

    if F_in_R <= 0:
        yield_val = 0.0
    else:
        yield_val = (produced_P / stoich_product) / (F_in_R / stoich_reactant)

    return conversion, selectivity, yield_val


if __name__ == "__main__":
    input_config = {
        "R0401": [
            ("TEMP", 100),
            ("RES_TIME", 4)
        ],

        "T0403": [
            ("NSTAGE", 6),
            ("BASIS_RR", 0.1),
            ("FEED_STAGE", "0410", 2)
        ],
    }

    output_config = {
        ("0410", "3-PN"): ["MOLEFLOW"],
        "R0401": "QCALC",
        "T0403": "DUTY",
        ("0411", "3-PN"): ["MOLEFRAC"]
    }
    aspen_path = r"test.bkp"
    result = run_aspen_with_structured_io(aspen_path, input_config=input_config, output_config=output_config)
    print(result)
    # connections_graph = extract_aspen_flowsheet_connections(aspen_path)
    # print(connections_graph)
