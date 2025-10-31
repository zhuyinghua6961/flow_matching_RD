<template>
  <div id="app" class="app-container">
    <!-- 顶部导航栏 -->
    <el-header class="app-header">
      <div class="header-content">
        <h1 class="title">
          <el-icon><PictureFilled /></el-icon>
          Sim2Real 推理 WebUI
        </h1>
        <div class="header-right">
          <el-button 
            type="info" 
            plain
            @click="showPluginManager = true"
            style="margin-right: 15px;"
          >
            <el-icon><Setting /></el-icon>
            插件管理
          </el-button>
          <ModelSelector />
        </div>
      </div>
    </el-header>

    <!-- 主内容区域 -->
    <el-container class="main-container">
      <!-- 左侧：上传和配置 -->
      <el-aside width="400px" class="left-panel">
        <el-card class="panel-card">
          <template #header>
            <span><el-icon><Upload /></el-icon> 图片上传</span>
          </template>
          <ImageUpload />
        </el-card>

        <el-card class="panel-card" style="margin-top: 20px;">
          <template #header>
            <span><el-icon><Setting /></el-icon> 推理参数</span>
          </template>
          <InferenceParams />
        </el-card>
      </el-aside>

      <!-- 右侧：结果展示 -->
      <el-main class="right-panel">
        <el-card class="result-card">
          <template #header>
            <span><el-icon><View /></el-icon> 推理结果</span>
          </template>
          <ResultDisplay />
        </el-card>
      </el-main>
    </el-container>

    <!-- 插件管理对话框 -->
    <PluginManager v-model="showPluginManager" />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import ImageUpload from './components/ImageUpload.vue'
import ResultDisplay from './components/ResultDisplay.vue'
import ModelSelector from './components/ModelSelector.vue'
import InferenceParams from './components/InferenceParams.vue'
import PluginManager from './components/PluginManager.vue'
import { useModelStore } from './stores/model'

const modelStore = useModelStore()
const showPluginManager = ref(false)

onMounted(() => {
  // 初始化：加载插件列表
  modelStore.loadPluginList()
})
</script>

<style scoped>
.app-container {
  width: 100vw;
  height: 100vh;
  background: #f5f7fa;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  height: 70px;
  display: flex;
  align-items: center;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

.title {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-right {
  min-width: 300px;
}

.main-container {
  flex: 1;
  padding: 20px;
  gap: 20px;
  overflow: hidden;
}

.left-panel {
  background: transparent;
  overflow-y: auto;
}

.right-panel {
  background: transparent;
  padding: 0;
  overflow-y: auto;
}

.panel-card,
.result-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.result-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* 滚动条样式 */
:deep(.el-scrollbar__wrap) {
  overflow-x: hidden;
}
</style>

