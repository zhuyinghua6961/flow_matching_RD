<template>
  <el-dialog
    v-model="visible"
    title="插件管理"
    width="800px"
    :close-on-click-modal="false"
  >
    <el-tabs v-model="activeTab">
      <!-- Tab 1: 已注册插件列表 -->
      <el-tab-pane label="已注册插件" name="list">
        <el-table 
          :data="modelStore.plugins" 
          style="width: 100%"
          v-loading="modelStore.loading"
        >
          <el-table-column prop="name" label="插件名称" width="180" />
          <el-table-column label="状态" width="120">
            <template #default="{ row }">
              <el-tag v-if="row.is_loaded" type="success">已加载</el-tag>
              <el-tag v-else type="info">未加载</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="当前使用" width="100" align="center">
            <template #default="{ row }">
              <el-icon v-if="row.is_current" color="#67c23a" :size="20">
                <Check />
              </el-icon>
            </template>
          </el-table-column>
          <el-table-column label="操作" align="center" width="200">
            <template #default="{ row }">
              <el-button-group>
                <el-button 
                  size="small" 
                  @click="handleViewInfo(row.name)"
                >
                  详情
                </el-button>
                <el-button 
                  v-if="!row.is_current"
                  size="small" 
                  type="primary"
                  @click="handleSwitch(row.name)"
                >
                  切换
                </el-button>
                <el-button 
                  v-if="row.is_loaded"
                  size="small" 
                  type="warning"
                  @click="handleUnloadModel(row.name)"
                  :loading="unloadingModel === row.name"
                >
                  卸载模型
                </el-button>
                <el-button 
                  size="small" 
                  type="danger"
                  @click="handleUnregister(row.name)"
                >
                  删除
                </el-button>
              </el-button-group>
            </template>
          </el-table-column>
        </el-table>

        <el-empty v-if="modelStore.plugins.length === 0" description="暂无插件" />
      </el-tab-pane>

      <!-- Tab 2: 注册新插件 -->
      <el-tab-pane label="注册新插件" name="register">
        <el-form :model="registerForm" label-width="120px">
          <el-form-item label="插件文件">
            <!-- 切换上传方式 -->
            <el-radio-group v-model="uploadMode" style="margin-bottom: 15px;">
              <el-radio label="upload">上传文件</el-radio>
              <el-radio label="path">指定路径</el-radio>
            </el-radio-group>

            <!-- 方式1: 上传文件 -->
            <el-upload
              v-if="uploadMode === 'upload'"
              ref="uploadRef"
              :auto-upload="false"
              :limit="1"
              :on-change="handlePluginFileChange"
              accept=".py"
              drag
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">
                拖拽插件文件到此处 或 <em>点击上传</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  只支持 .py 文件，需继承 InferenceInterface
                </div>
              </template>
            </el-upload>

            <!-- 方式2: 指定路径 -->
            <div v-else>
              <el-input
                v-model="registerForm.plugin_file_path"
                placeholder="例如: /home/user/桌面/flow_matching_RD/webui/backend/plugins/flow_matching_v2_plugin.py"
              >
                <template #prepend>
                  <el-icon><Folder /></el-icon>
                </template>
              </el-input>
              <div class="form-tip">
                输入服务器上的插件文件绝对路径或相对路径（相对于 backend/ 目录）
              </div>
            </div>
          </el-form-item>

          <el-divider />

          <el-form-item label="插件类名" required>
            <el-input 
              v-model="registerForm.plugin_class_name" 
              placeholder="例如: FlowMatchingV2Plugin"
            >
              <template #prepend>class</template>
            </el-input>
            <div class="form-tip">插件文件中定义的类名</div>
          </el-form-item>

          <el-form-item label="插件注册名" required>
            <el-input 
              v-model="registerForm.plugin_name" 
              placeholder="例如: flow_matching_v2"
            >
              <template #prepend>name</template>
            </el-input>
            <div class="form-tip">用于标识插件的唯一名称</div>
          </el-form-item>

          <el-divider>配置参数</el-divider>

          <el-form-item label="模型路径" required>
            <el-input 
              v-model="registerForm.config.checkpoint_path" 
              placeholder="例如: /home/user/桌面/flow_matching_RD/outputs_v2/checkpoints/best_model.pth"
            >
              <template #prepend>
                <el-icon><Folder /></el-icon>
              </template>
            </el-input>
            <div class="form-tip">
              输入服务器上的模型文件绝对路径（.pth, .pt, .ckpt等）
            </div>
          </el-form-item>

          <el-form-item label="设备">
            <el-select v-model="registerForm.config.device">
              <el-option label="CUDA:0 (GPU 0)" value="cuda:0" />
              <el-option label="CUDA:1 (GPU 1)" value="cuda:1" />
              <el-option label="CPU" value="cpu" />
            </el-select>
          </el-form-item>

          <el-form-item label="自定义参数">
            <el-button 
              type="primary" 
              link 
              @click="showCustomParamsDialog = true"
            >
              <el-icon><Setting /></el-icon>
              配置自定义参数 (JSON)
            </el-button>
            <div class="form-tip">
              根据你的插件需要添加额外参数（如 base_channels, image_size 等）
            </div>
          </el-form-item>
        </el-form>

        <div class="dialog-footer">
          <el-button @click="handleResetForm">重置</el-button>
          <el-button 
            type="primary" 
            @click="handleRegister"
            :loading="registering"
            :disabled="!canRegister"
          >
            注册插件
          </el-button>
        </div>
      </el-tab-pane>

      <!-- Tab 3: 插件模板 -->
      <el-tab-pane label="开发指南" name="guide">
        <el-card>
          <template #header>
            <span>📝 如何开发插件</span>
          </template>
          
          <el-steps :active="4" finish-status="success" simple>
            <el-step title="下载模板" />
            <el-step title="实现接口" />
            <el-step title="上传插件" />
            <el-step title="注册使用" />
          </el-steps>

          <el-divider />

          <div class="guide-content">
            <h4>1. 下载插件模板</h4>
            <el-button type="primary" @click="downloadTemplate">
              <el-icon><Download /></el-icon>
              下载 plugin_template.py
            </el-button>

            <h4 style="margin-top: 20px;">2. 实现必需方法</h4>
            <ul>
              <li><code>load_model()</code> - 加载模型</li>
              <li><code>unload_model()</code> - 卸载模型</li>
              <li><code>inference()</code> - 单张推理</li>
              <li><code>batch_inference()</code> - 批量推理</li>
              <li><code>get_model_info()</code> - 获取模型信息</li>
            </ul>

            <h4>3. 上传并注册</h4>
            <p>切换到"注册新插件"标签页，上传你的插件文件。</p>

            <el-alert type="info" :closable="false" style="margin-top: 15px;">
              <template #title>
                详细开发指南请参考项目文档 PLUGIN_GUIDE.md
              </template>
            </el-alert>
          </div>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <!-- 自定义参数对话框 -->
    <el-dialog
      v-model="showCustomParamsDialog"
      title="自定义参数 (JSON格式)"
      width="600px"
      append-to-body
    >
      <el-input
        v-model="customParamsJson"
        type="textarea"
        :rows="10"
        placeholder='例如:
{
  "base_channels": 64,
  "channel_mult": [1, 2, 4, 8],
  "attention_levels": [],
  "image_size": [512, 512]
}'
      />
      <template #footer>
        <el-button @click="showCustomParamsDialog = false">取消</el-button>
        <el-button type="primary" @click="handleSaveCustomParams">
          保存
        </el-button>
      </template>
    </el-dialog>

    <!-- 插件详情对话框 -->
    <el-dialog
      v-model="showInfoDialog"
      title="插件详情"
      width="600px"
      append-to-body
    >
      <el-descriptions v-if="currentPluginInfo" :column="1" border>
        <el-descriptions-item label="插件名称">
          {{ currentPluginInfo.plugin_name }}
        </el-descriptions-item>
        <el-descriptions-item label="加载状态">
          <el-tag v-if="currentPluginInfo.is_loaded" type="success">已加载</el-tag>
          <el-tag v-else type="info">未加载</el-tag>
        </el-descriptions-item>
        <el-descriptions-item 
          v-if="currentPluginInfo.model_info"
          label="模型名称"
        >
          {{ currentPluginInfo.model_info.name }}
        </el-descriptions-item>
        <el-descriptions-item 
          v-if="currentPluginInfo.model_info"
          label="版本"
        >
          {{ currentPluginInfo.model_info.version }}
        </el-descriptions-item>
        <el-descriptions-item 
          v-if="currentPluginInfo.model_info"
          label="描述"
        >
          {{ currentPluginInfo.model_info.description }}
        </el-descriptions-item>
        <el-descriptions-item 
          v-if="currentPluginInfo.model_info && currentPluginInfo.model_info.parameters"
          label="参数量"
        >
          {{ (currentPluginInfo.model_info.parameters / 1e6).toFixed(1) }}M
        </el-descriptions-item>
      </el-descriptions>
    </el-dialog>
  </el-dialog>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useModelStore } from '@/stores/model'
import { ElMessage, ElMessageBox } from 'element-plus'
import { 
  uploadPlugin, 
  registerPlugin, 
  unregisterPlugin,
  unloadModel,
  getPluginInfo 
} from '@/api/model'

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue'])

const modelStore = useModelStore()

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})

const activeTab = ref('list')
const registering = ref(false)
const uploadRef = ref(null)
const showCustomParamsDialog = ref(false)
const showInfoDialog = ref(false)
const currentPluginInfo = ref(null)
const customParamsJson = ref('')
const uploadMode = ref('path') // 'upload' 或 'path'
const unloadingModel = ref(null) // 正在卸载模型的插件名称

const registerForm = ref({
  plugin_file: null,
  plugin_file_path: '', // 新增：文件路径
  plugin_class_name: '',
  plugin_name: '',
  config: {
    checkpoint_path: '',
    device: 'cuda:0'
  }
})

const canRegister = computed(() => {
  const hasPluginFile = uploadMode.value === 'upload' 
    ? registerForm.value.plugin_file !== null
    : registerForm.value.plugin_file_path.trim() !== ''
  
  return hasPluginFile &&
         registerForm.value.plugin_class_name &&
         registerForm.value.plugin_name &&
         registerForm.value.config.checkpoint_path
})

const handlePluginFileChange = (file) => {
  registerForm.value.plugin_file = file.raw
  ElMessage.success(`已选择: ${file.name}`)
}

const handleSaveCustomParams = () => {
  try {
    const params = JSON.parse(customParamsJson.value)
    registerForm.value.config = {
      ...registerForm.value.config,
      ...params
    }
    showCustomParamsDialog.value = false
    ElMessage.success('自定义参数已保存')
  } catch (e) {
    ElMessage.error('JSON格式错误，请检查')
  }
}

const handleRegister = async () => {
  registering.value = true
  
  try {
    let pluginFilePath = ''
    
    if (uploadMode.value === 'upload') {
      // 模式1: 上传文件
      const uploadResult = await uploadPlugin(registerForm.value.plugin_file)
      
      if (!uploadResult.success) {
        ElMessage.error(uploadResult.message || '上传插件失败')
        return
      }
      
      pluginFilePath = uploadResult.file_path
    } else {
      // 模式2: 使用指定路径
      pluginFilePath = registerForm.value.plugin_file_path
    }
    
    // 注册插件
    const registerResult = await registerPlugin({
      plugin_file: pluginFilePath,
      plugin_class_name: registerForm.value.plugin_class_name,
      plugin_name: registerForm.value.plugin_name,
      config: registerForm.value.config
    })
    
    if (registerResult.success) {
      ElMessage.success('插件注册成功！')
      handleResetForm()
      activeTab.value = 'list'
      await modelStore.loadPluginList()
    } else {
      ElMessage.error(registerResult.message || '注册失败')
    }
  } catch (error) {
    console.error(error)
    ElMessage.error('注册失败: ' + error.message)
  } finally {
    registering.value = false
  }
}

const handleResetForm = () => {
  registerForm.value = {
    plugin_file: null,
    plugin_file_path: '',
    plugin_class_name: '',
    plugin_name: '',
    config: {
      checkpoint_path: '',
      device: 'cuda:0'
    }
  }
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
  uploadMode.value = 'path' // 重置为路径模式
}

const handleSwitch = async (pluginName) => {
  await modelStore.switchToPlugin(pluginName)
}

const handleUnregister = async (pluginName) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除插件 "${pluginName}" 吗？`,
      '确认删除',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    const result = await unregisterPlugin(pluginName)
    if (result.success) {
      ElMessage.success('插件已删除')
      await modelStore.loadPluginList()
    } else {
      ElMessage.error(result.message || '删除失败')
    }
  } catch {
    // 用户取消
  }
}

const handleUnloadModel = async (pluginName) => {
  try {
    await ElMessageBox.confirm(
      `确定要卸载插件 "${pluginName}" 的模型吗？\n卸载后将释放显存，但插件仍然保持注册状态。`,
      '确认卸载模型',
      {
        confirmButtonText: '卸载',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    unloadingModel.value = pluginName
    
    const result = await unloadModel(pluginName)
    if (result.success) {
      ElMessage.success('模型卸载成功，显存已释放')
      await modelStore.loadPluginList() // 刷新插件列表状态
    } else {
      ElMessage.error(result.message || '模型卸载失败')
    }
  } catch {
    // 用户取消
  } finally {
    unloadingModel.value = null
  }
}

const handleViewInfo = async (pluginName) => {
  const info = await modelStore.getModelInfo(pluginName)
  if (info) {
    currentPluginInfo.value = info
    showInfoDialog.value = true
  }
}

const downloadTemplate = () => {
  // 下载插件模板
  window.open('/api/static/plugin_template.py', '_blank')
  ElMessage.success('模板下载已开始')
}
</script>

<style scoped>
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
  line-height: 1.4;
}

.guide-content {
  padding: 20px 0;
}

.guide-content h4 {
  margin: 15px 0 10px 0;
  color: #303133;
}

.guide-content ul {
  margin: 10px 0;
  padding-left: 20px;
}

.guide-content li {
  margin: 8px 0;
  line-height: 1.6;
}

.guide-content code {
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 3px;
  font-family: monospace;
  color: #e96900;
}
</style>

