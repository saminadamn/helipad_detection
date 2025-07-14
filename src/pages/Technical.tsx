import React from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, 
  Layers, 
  Settings, 
  BarChart3, 
  Code, 
  Database,
  Cpu,
  Zap,
  Target,
  Clock
} from 'lucide-react'

const Technical = () => {
  const architectureLayers = [
    {
      name: 'Input Layer',
      description: '224×224×3 RGB images',
      details: 'Normalized pixel values [0,1]'
    },
    {
      name: 'Conv2D (32 filters)',
      description: '3×3 kernels + ReLU + MaxPool',
      details: 'Feature extraction with dropout (0.25)'
    },
    {
      name: 'Conv2D (64 filters)',
      description: '3×3 kernels + ReLU + MaxPool',
      details: 'Deeper feature learning with dropout (0.25)'
    },
    {
      name: 'Conv2D (128 filters)',
      description: '3×3 kernels + ReLU + MaxPool',
      details: 'Complex pattern recognition with dropout (0.25)'
    },
    {
      name: 'Global Average Pooling',
      description: 'Spatial dimension reduction',
      details: 'Prevents overfitting, reduces parameters'
    },
    {
      name: 'Dense (512 neurons)',
      description: 'ReLU activation + Dropout (0.5)',
      details: 'High-level feature combination'
    },
    {
      name: 'Output Layer',
      description: 'Dense (2) + Softmax',
      details: 'Binary classification: [No Helipad, Helipad]'
    }
  ]

  const trainingConfig = [
    { label: 'Optimizer', value: 'Adam (lr=0.001)', icon: Settings },
    { label: 'Loss Function', value: 'Sparse Categorical Crossentropy', icon: Target },
    { label: 'Batch Size', value: '16', icon: Database },
    { label: 'Epochs', value: '20', icon: Clock },
    { label: 'Image Size', value: '224×224 pixels', icon: Layers },
    { label: 'Dataset Split', value: '80% train, 20% test', icon: BarChart3 }
  ]

  const performanceMetrics = [
    { metric: 'Test Accuracy', value: '92.5%', color: 'text-green-600' },
    { metric: 'Training Accuracy', value: '95.2%', color: 'text-blue-600' },
    { metric: 'Validation Accuracy', value: '91.8%', color: 'text-purple-600' },
    { metric: 'Inference Time', value: '<100ms', color: 'text-yellow-600' }
  ]

  const datasetInfo = [
    { category: 'Real Helipad Images', count: '200+', description: 'Actual photographs from various sources' },
    { category: 'Augmented Helipads', count: '300', description: 'Rotation, brightness, contrast variations' },
    { category: 'Negative Samples', count: '300', description: 'Urban, rural, water, and random scenes' },
    { category: 'Total Training Data', count: '600+', description: 'Balanced dataset for robust training' }
  ]

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Technical Architecture
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Deep dive into the CNN architecture, training methodology, and performance optimization 
              techniques used in our helipad detection system.
            </p>
          </motion.div>
        </div>

        {/* Model Architecture */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Brain className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">CNN Architecture</h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Architecture Diagram */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Sequential Model Layers</h3>
                <div className="space-y-3">
                  {architectureLayers.map((layer, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                      className="bg-gray-50 p-4 rounded-lg border-l-4 border-primary-500"
                    >
                      <div className="font-semibold text-gray-900">{layer.name}</div>
                      <div className="text-gray-700 text-sm mt-1">{layer.description}</div>
                      <div className="text-gray-500 text-xs mt-1">{layer.details}</div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Architecture Benefits */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Design Rationale</h3>
                <div className="space-y-6">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-blue-900 mb-2">Progressive Feature Extraction</h4>
                    <p className="text-blue-800 text-sm">
                      32→64→128 filter progression captures increasingly complex patterns, 
                      from edges and textures to complete helipad structures.
                    </p>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-green-900 mb-2">Regularization Strategy</h4>
                    <p className="text-green-800 text-sm">
                      Dropout layers (0.25-0.5) prevent overfitting with limited data, 
                      ensuring good generalization to unseen images.
                    </p>
                  </div>
                  
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-purple-900 mb-2">Optimal Depth</h4>
                    <p className="text-purple-800 text-sm">
                      3 convolutional blocks balance model complexity with training stability, 
                      achieving 92.5% accuracy on real-world data.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Training Configuration */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Settings className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Training Configuration</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {trainingConfig.map((config, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-gray-50 p-6 rounded-lg text-center"
                >
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mb-4">
                    <config.icon className="h-6 w-6 text-primary-600" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-2">{config.label}</h3>
                  <p className="text-gray-600 font-mono text-sm">{config.value}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Performance Metrics */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <BarChart3 className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Performance Metrics</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {performanceMetrics.map((metric, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-white p-6 rounded-lg border-2 border-gray-100 text-center"
                >
                  <div className={`text-3xl font-bold mb-2 ${metric.color}`}>
                    {metric.value}
                  </div>
                  <div className="text-gray-600 font-medium">{metric.metric}</div>
                </motion.div>
              ))}
            </div>

            {/* Classification Report */}
            <div className="bg-gray-900 text-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4 text-green-400">Classification Report</h3>
              <div className="font-mono text-sm space-y-1">
                <div className="grid grid-cols-5 gap-4 border-b border-gray-700 pb-2 mb-2">
                  <div></div>
                  <div className="text-center">precision</div>
                  <div className="text-center">recall</div>
                  <div className="text-center">f1-score</div>
                  <div className="text-center">support</div>
                </div>
                <div className="grid grid-cols-5 gap-4">
                  <div>No Helipad</div>
                  <div className="text-center">0.94</div>
                  <div className="text-center">0.93</div>
                  <div className="text-center">0.93</div>
                  <div className="text-center">62</div>
                </div>
                <div className="grid grid-cols-5 gap-4">
                  <div>Helipad</div>
                  <div className="text-center">0.93</div>
                  <div className="text-center">0.91</div>
                  <div className="text-center">0.92</div>
                  <div className="text-center">58</div>
                </div>
                <div className="grid grid-cols-5 gap-4 border-t border-gray-700 pt-2 mt-2">
                  <div>accuracy</div>
                  <div></div>
                  <div></div>
                  <div className="text-center">0.93</div>
                  <div className="text-center">120</div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Dataset Information */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Database className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Dataset & Training Process</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Dataset Composition */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Dataset Composition</h3>
                <div className="space-y-4">
                  {datasetInfo.map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                      className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                    >
                      <div>
                        <div className="font-semibold text-gray-900">{item.category}</div>
                        <div className="text-sm text-gray-600">{item.description}</div>
                      </div>
                      <div className="text-2xl font-bold text-primary-600">{item.count}</div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Data Augmentation */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Data Augmentation Strategy</h3>
                <div className="space-y-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-blue-900 mb-2">Geometric Transformations</h4>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• Rotation: ±45 degrees (helipads from any angle)</li>
                      <li>• Horizontal/Vertical flipping</li>
                      <li>• Random cropping and resizing</li>
                    </ul>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-green-900 mb-2">Photometric Variations</h4>
                    <ul className="text-green-800 text-sm space-y-1">
                      <li>• Brightness: 0.7x to 1.3x (lighting conditions)</li>
                      <li>• Contrast adjustment (weather variations)</li>
                      <li>• Gaussian noise injection (image quality)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Implementation Details */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Code className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Implementation Details</h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Code Example */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Model Architecture Code</h3>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                  <pre className="text-sm">
{`model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    # First block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Classifier
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)`}
                  </pre>
                </div>
              </div>

              {/* Technical Specifications */}
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Technical Specifications</h3>
                <div className="space-y-4">
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <Cpu className="h-5 w-5 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Framework</div>
                      <div className="text-sm text-gray-600">TensorFlow 2.x / Keras</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <Zap className="h-5 w-5 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Hardware Requirements</div>
                      <div className="text-sm text-gray-600">CPU-only, 4GB RAM minimum</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <Target className="h-5 w-5 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Input Format</div>
                      <div className="text-sm text-gray-600">224×224 RGB images, normalized [0,1]</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <Clock className="h-5 w-5 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Inference Speed</div>
                      <div className="text-sm text-gray-600"><100ms per image on CPU</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Future Enhancements */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8 bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200"
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-6">Future Enhancements</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Planned Features</h3>
                <ul className="space-y-2 text-gray-700">
                  <li>• Multi-class classification (hospital, private, military helipads)</li>
                  <li>• Object detection with bounding box coordinates</li>
                  <li>• Real-time video stream processing</li>
                  <li>• Mobile deployment optimization</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Technical Roadmap</h3>
                <ul className="space-y-2 text-gray-700">
                  <li>• Transfer learning with pre-trained models (ResNet, EfficientNet)</li>
                  <li>• Semantic segmentation for pixel-level detection</li>
                  <li>• Multi-modal input (RGB + infrared/thermal)</li>
                  <li>• 3D analysis and landing suitability assessment</li>
                </ul>
              </div>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
  )
}

export default Technical