# IP访问配置说明

本文档说明如何通过IP地址 `172.18.8.31` 访问WebUI。

---

## ✅ 已完成的修改

### 1. 后端CORS配置 (`webui/backend/config.py`)

**修改内容**:
```python
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://172.18.8.31:5173",  # ← 添加服务器IP
    "http://172.18.8.31:3000",
]
```

**作用**: 允许来自 `http://172.18.8.31:5173` 的跨域请求

---

### 2. 前端API配置 (`webui/frontend/src/components/ModelSelector.vue`)

**修改内容**:
```javascript
// 修改前
const API_BASE_URL = 'http://localhost:8000'

// 修改后（使用相对路径）
const API_BASE_URL = ''
```

**作用**: 
- 请求会变成相对路径 `/api/models/scanned`
- 自动通过Vite代理转发到后端
- 支持IP访问

---

## 🚀 启动服务

### 1. 启动后端

```bash
cd /home/user/桌面/flow_matching_RD/webui/backend
python main.py
```

**确认启动成功**:
```
✓ 自动切换工作目录到项目根目录
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**注意**: `0.0.0.0:8000` 表示监听所有网络接口，可以通过IP访问

---

### 2. 启动前端

```bash
cd /home/user/桌面/flow_matching_RD/webui/frontend
npm run dev
```

**确认启动成功**:
```
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://172.18.8.31:5173/
```

**注意**: Vite会自动检测并显示网络地址

---

## 🌐 访问方式

### 本机访问（服务器上的浏览器）
```
http://localhost:5173
http://127.0.0.1:5173
```

### 远程访问（其他机器的浏览器）
```
http://172.18.8.31:5173
```

**完整流程**:
```
浏览器访问: http://172.18.8.31:5173
  ↓
前端Vite服务: 172.18.8.31:5173
  ↓
前端发起API请求: /api/models/scanned (相对路径)
  ↓
Vite代理转发: localhost:8000/api/models/scanned
  ↓
后端FastAPI: 0.0.0.0:8000 (监听所有接口)
  ↓
CORS检查: Origin = http://172.18.8.31:5173 ✅ 在允许列表
  ↓
返回数据 ✅
```

---

## 🔧 端口和防火墙

### 需要开放的端口

**后端**: `8000`
```bash
# 检查端口是否开放
sudo firewall-cmd --list-ports

# 临时开放（立即生效）
sudo firewall-cmd --add-port=8000/tcp

# 永久开放
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

**前端**: `5173`
```bash
# 临时开放
sudo firewall-cmd --add-port=5173/tcp

# 永久开放
sudo firewall-cmd --add-port=5173/tcp --permanent
sudo firewall-cmd --reload
```

### 验证端口监听

```bash
# 检查端口是否在监听
netstat -tulpn | grep 8000
netstat -tulpn | grep 5173

# 或使用ss命令
ss -tulpn | grep 8000
ss -tulpn | grep 5173
```

**期望输出**:
```
tcp   0   0 0.0.0.0:8000   0.0.0.0:*   LISTEN   12345/python
tcp   0   0 0.0.0.0:5173   0.0.0.0:*   LISTEN   12346/node
```

---

## 🐛 故障排查

### 问题1: 无法访问前端页面

**症状**: 浏览器访问 `http://172.18.8.31:5173` 无法打开

**排查**:
```bash
# 1. 检查前端是否启动
ps aux | grep "vite\|npm"

# 2. 检查端口是否监听
ss -tulpn | grep 5173

# 3. 检查防火墙
sudo firewall-cmd --list-ports

# 4. 测试本地访问
curl http://localhost:5173
```

**解决**:
- 确保前端已启动: `npm run dev`
- 确保vite.config.js中 `host: '0.0.0.0'` ✅ 已配置
- 开放5173端口

---

### 问题2: 前端加载成功，但API调用失败

**症状**: 页面显示，但"扫描模型"、"加载模型"失败

**排查**:
```bash
# 1. 检查后端是否启动
ps aux | grep "python.*main.py"

# 2. 检查后端端口
ss -tulpn | grep 8000

# 3. 在服务器上测试后端
curl http://localhost:8000/api/models/scanned

# 4. 查看后端日志
# 在后端终端查看是否有错误输出
```

**浏览器检查**:
1. 打开浏览器开发者工具 (F12)
2. 切换到 "Network" (网络) 标签
3. 点击"扫描模型"按钮
4. 查看请求状态:
   - **200 OK**: 成功 ✅
   - **403 Forbidden**: CORS问题 → 检查后端config.py
   - **404 Not Found**: 路由问题 → 检查API路径
   - **Failed to fetch**: 网络问题 → 检查防火墙

---

### 问题3: CORS错误

**症状**: 浏览器控制台显示类似错误:
```
Access to XMLHttpRequest at 'http://172.18.8.31:8000/api/...' 
from origin 'http://172.18.8.31:5173' has been blocked by CORS policy
```

**解决**:
1. 确认 `webui/backend/config.py` 中已添加:
   ```python
   "http://172.18.8.31:5173"
   ```

2. 重启后端服务（修改config.py后必须重启）:
   ```bash
   # 停止旧进程
   pkill -f "python.*main.py"
   
   # 重新启动
   cd /home/user/桌面/flow_matching_RD/webui/backend
   python main.py
   ```

---

### 问题4: 从其他机器访问时"连接被拒绝"

**症状**: 从另一台电脑访问 `http://172.18.8.31:5173` 失败

**排查**:
```bash
# 在其他机器上测试连通性
ping 172.18.8.31
telnet 172.18.8.31 5173
telnet 172.18.8.31 8000
```

**可能原因**:
1. 防火墙未开放端口
2. 服务只监听了localhost而不是0.0.0.0
3. 网络隔离（不同子网/VLAN）

---

## 📊 配置文件总览

### `webui/backend/config.py`
```python
HOST = "0.0.0.0"         # ✅ 监听所有网络接口
PORT = 8000              # ✅ 后端端口

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://172.18.8.31:5173",  # ✅ 允许IP访问
]
```

### `webui/frontend/vite.config.js`
```javascript
server: {
  host: '0.0.0.0',       // ✅ 监听所有网络接口
  port: 5173,            // ✅ 前端端口
  proxy: {
    '/api': {
      target: 'http://localhost:8000',  // ✅ Vite代理目标
      changeOrigin: true
    }
  }
}
```

### `webui/frontend/src/components/ModelSelector.vue`
```javascript
const API_BASE_URL = ''  // ✅ 使用相对路径（走代理）
```

---

## ✅ 验证清单

访问成功的标志:

- [ ] 后端启动成功，显示 `Uvicorn running on http://0.0.0.0:8000`
- [ ] 前端启动成功，显示 `Network: http://172.18.8.31:5173/`
- [ ] 防火墙已开放 8000 和 5173 端口
- [ ] 浏览器访问 `http://172.18.8.31:5173` 可以看到页面
- [ ] 点击"扫描模型"可以看到模型列表
- [ ] 选择模型后点击"加载模型"可以成功加载
- [ ] 浏览器控制台(F12)没有CORS或网络错误

---

## 🎉 完成

修改完成后，重启前后端服务，即可通过IP地址访问WebUI！

**访问地址**: http://172.18.8.31:5173

