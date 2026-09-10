"""平台统一轨迹字段及常见实车数据列名别名。"""

# 所有预处理结果都按此顺序输出，保证 CIA、EEA 和 PES 对字段含义理解一致。
STANDARD_TRAJECTORY_COLUMNS = [
    "longitude",
    "latitude",
    "time",
    "speed",
    "direction",
    "forward_reverse_label",
]

# 兼容英文、带单位表头和中文表头，避免用户为不同采集设备手工改脚本。
ALIASES = {
    "longitude": ["longitude", "Longitude (°)", "lon", "Lon", "经度"],
    "latitude": ["latitude", "Latitude (°)", "lat", "Lat", "纬度"],
    "time": ["time", "Time", "timestamp", "Timestamp", "时间"],
    "speed": ["speed", "Speed", "speed_kmh", "速度"],
    "direction": ["direction", "Direction", "heading", "Heading", "航向"],
    "forward_reverse_label": ["forward_reverse_label", "State", "state", "direction_label"],
}
