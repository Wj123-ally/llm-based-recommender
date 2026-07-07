#!/usr/bin/env python3
"""诊断 Docker UI 端口映射问题"""

import subprocess
import json

print("=" * 70)
print("Docker UI 端口映射诊断")
print("=" * 70)

# 1. 检查容器配置
print("\n1. 容器端口绑定配置 (HostConfig.PortBindings):")
result = subprocess.run(
    ["docker", "inspect", "recommender-ui", "--format", "{{json .HostConfig.PortBindings}}"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    data = json.loads(result.stdout)
    print(json.dumps(data, indent=2))
else:
    print(f"错误: {result.stderr}")

# 2. 检查实际网络端口
print("\n2. 实际网络端口状态 (NetworkSettings.Ports):")
result = subprocess.run(
    ["docker", "inspect", "recommender-ui", "--format", "{{json .NetworkSettings.Ports}}"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    data = json.loads(result.stdout)
    print(json.dumps(data, indent=2))
else:
    print(f"错误: {result.stderr}")

# 3. 检查容器网络模式
print("\n3. 容器网络模式:")
result = subprocess.run(
    ["docker", "inspect", "recommender-ui", "--format", "{{.HostConfig.NetworkMode}}"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print(result.stdout.strip())

# 4. 检查容器状态
print("\n4. 容器状态:")
result = subprocess.run(
    ["docker", "inspect", "recommender-ui", "--format", "{{.State.Status}}"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print(result.stdout.strip())

# 5. 检查 docker-compose 配置
print("\n5. docker-compose 解析后的 UI 配置:")
result = subprocess.run(
    ["docker-compose", "config"],
    capture_output=True,
    text=True,
    cwd="D:\\github_project\\简历项目\\llm-based-recommender"
)
if result.returncode == 0:
    lines = result.stdout.split('\n')
    in_ui_section = False
    for i, line in enumerate(lines):
        if 'ui:' in line and not line.strip().startswith('#'):
            in_ui_section = True
            print(line)
        elif in_ui_section:
            if line and not line.startswith(' ') and ':' in line:
                break
            print(line)
            if 'ports:' in line:
                # 打印 ports 后面的几行
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip() and not lines[j].startswith(' ' * 6):
                        break
                    print(lines[j])

# 6. 检查环境变量
print("\n6. 环境变量 UI_PORT:")
result = subprocess.run(
    ["docker", "inspect", "recommender-ui", "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    for line in result.stdout.split('\n'):
        if 'UI_PORT' in line or 'API_URL' in line:
            print(line)

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
