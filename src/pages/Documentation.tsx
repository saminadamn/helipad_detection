import { motion } from 'framer-motion'
import { 
  Book, 
  Download, 
  Terminal, 
  Play, 
  Settings, 
  FileText,
  Code,
  Lightbulb,
  ExternalLink,
  Copy,
  CheckCircle
} from 'lucide-react'

const Documentation = () => {
  const [copiedCode, setCopiedCode] = React.useState<string | null>(null)

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const installationSteps = [
    {
      title: 'Clone Repository',
      code: 'git clone https://github.com/saminadamn/helipad-detection.git\ncd helipad-detection',
      description: 'Download the project from GitHub'
    },
    {
      title: 'Create Virtual Environment',
      code: 'python -m venv venv\n# Windows:\nvenv\\Scripts\\activate\n# Linux/Mac:\nsource venv/bin/activate',
      description: 'Set up isolated Python environment'
    },
    {
      title: 'Install Dependencies',
      code: 'pip install -r requirements.txt',
      description: 'Install required Python packages'
    },
    {
      title: 'Test Installation',
      code: 'python src/predict.py Aug_Illustration.PNG',
      description: 'Run a quick test with sample image'
    }
  ]

  const usageExamples = [
    {
      title: 'Single Image Prediction',
      code: `python src/predict.py path/to/your/image.jpg

# Expected output:
# 🎯 Prediction: Helipad
# 📊 Confidence: 94.2%
# 📈 Probabilities: No Helipad: 0.058, Helipad: 0.942`,
      description: 'Analyze a single image for helipad detection'
    },
    {
      title: 'Batch Processing',
      code: `import tensorflow as tf
from src.predict import predict_helipad

# Load model once
model = tf.keras.models.load_model('models/helipad_classifier.h5')

# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
for img_path in image_paths:
    result, confidence = predict_helipad(img_path)
    print(f"{img_path}: {'Helipad' if result else 'No Helipad'} ({confidence:.1%})")`,
      description: 'Process multiple images efficiently'
    },
    {
      title: 'Training Your Own Model',
      code: `# Prepare your helipad images in data_image/ directory
python src/train.py

# Output:
# 📂 Loading images from: data_image
# 📊 Found 200 image files
# ✅ Loaded 200 helipad images
# 🔄 Creating 300 augmented images...
# 🎯 Test Accuracy: 0.9250 (92.50%)
# ✅ Model saved: helipad_classifier.h5`,
      description: 'Train the model with your own dataset'
    }
  ]

  const apiReference = [
    {
      function: 'predict_helipad(image_path)',
      description: 'Main prediction function for single images',
      parameters: [
        { name: 'image_path', type: 'str', description: 'Path to the image file' }
      ],
      returns: 'tuple: (prediction: bool, confidence: float)'
    },
    {
      function: 'load_and_preprocess_image(img_path)',
      description: 'Load and preprocess image for model input',
      parameters: [
        { name: 'img_path', type: 'str', description: 'Path to the image file' }
      ],
      returns: 'numpy.ndarray: Preprocessed image array'
    },
    {
      function: 'create_model()',
      description: 'Create the CNN model architecture',
      parameters: [],
      returns: 'tensorflow.keras.Model: Compiled model'
    }
  ]

  const troubleshooting = [
    {
      issue: 'ModuleNotFoundError: No module named \'tensorflow\'',
      solution: 'Install TensorFlow: pip install tensorflow',
      category: 'Installation'
    },
    {
      issue: 'Model file not found error',
      solution: 'Run training first: python src/train.py, or download pre-trained model',
      category: 'Model'
    },
    {
      issue: 'Low accuracy on custom images',
      solution: 'Ensure images are aerial/satellite view, good quality, and similar to training data',
      category: 'Performance'
    },
    {
      issue: 'Out of memory error during training',
      solution: 'Reduce batch size in training configuration or use smaller image size',
      category: 'Training'
    }
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
              Documentation
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Complete guide to installation, usage, and API reference for the helipad detection system. 
              Get started quickly with step-by-step instructions and code examples.
            </p>
          </motion.div>
        </div>

        {/* Quick Start */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Play className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Quick Start</h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Prerequisites</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span>Python 3.8 or higher</span>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span>4GB+ RAM</span>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span>GPU recommended (optional)</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Quick Demo</h3>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-green-400 text-sm">Terminal</span>
                    <button
                      onClick={() => copyToClipboard('python src/predict.py Aug_Illustration.PNG', 'quick-demo')}
                      className="text-gray-400 hover:text-white transition-colors duration-200"
                    >
                      {copiedCode === 'quick-demo' ? (
                        <CheckCircle className="h-4 w-4 text-green-400" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  <pre className="text-sm">
{`# Test with sample image
python src/predict.py Aug_Illustration.PNG

# Expected output:
🎯 Prediction: Helipad
📊 Confidence: 94.2%`}
                  </pre>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Installation */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Download className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Installation</h2>
            </div>

            <div className="space-y-6">
              {installationSteps.map((step, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="border border-gray-200 rounded-lg p-6"
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="flex-shrink-0 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      {index + 1}
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900">{step.title}</h3>
                  </div>
                  
                  <p className="text-gray-600 mb-4">{step.description}</p>
                  
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-green-400 text-sm">Terminal</span>
                      <button
                        onClick={() => copyToClipboard(step.code, `install-${index}`)}
                        className="text-gray-400 hover:text-white transition-colors duration-200"
                      >
                        {copiedCode === `install-${index}` ? (
                          <CheckCircle className="h-4 w-4 text-green-400" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                    <pre className="text-sm whitespace-pre-wrap">{step.code}</pre>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Usage Examples */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Terminal className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Usage Examples</h2>
            </div>

            <div className="space-y-8">
              {usageExamples.map((example, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="border border-gray-200 rounded-lg p-6"
                >
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">{example.title}</h3>
                  <p className="text-gray-600 mb-4">{example.description}</p>
                  
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-blue-400 text-sm">Python</span>
                      <button
                        onClick={() => copyToClipboard(example.code, `usage-${index}`)}
                        className="text-gray-400 hover:text-white transition-colors duration-200"
                      >
                        {copiedCode === `usage-${index}` ? (
                          <CheckCircle className="h-4 w-4 text-green-400" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </button>
                    </div>
                    <pre className="text-sm whitespace-pre-wrap">{example.code}</pre>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* API Reference */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Code className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">API Reference</h2>
            </div>

            <div className="space-y-6">
              {apiReference.map((api, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="border border-gray-200 rounded-lg p-6"
                >
                  <div className="mb-4">
                    <h3 className="text-lg font-mono font-semibold text-primary-600 mb-2">
                      {api.function}
                    </h3>
                    <p className="text-gray-600">{api.description}</p>
                  </div>

                  {api.parameters.length > 0 && (
                    <div className="mb-4">
                      <h4 className="font-semibold text-gray-900 mb-2">Parameters:</h4>
                      <div className="space-y-2">
                        {api.parameters.map((param, paramIndex) => (
                          <div key={paramIndex} className="bg-gray-50 p-3 rounded">
                            <span className="font-mono text-sm text-primary-600">{param.name}</span>
                            <span className="text-gray-500 text-sm"> ({param.type})</span>
                            <span className="text-gray-700 text-sm"> - {param.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">Returns:</h4>
                    <div className="bg-gray-50 p-3 rounded">
                      <span className="font-mono text-sm text-green-600">{api.returns}</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Troubleshooting */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Settings className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Troubleshooting</h2>
            </div>

            <div className="space-y-4">
              {troubleshooting.map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="border border-gray-200 rounded-lg p-6"
                >
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0">
                      <span className={`inline-block px-2 py-1 text-xs font-medium rounded ${
                        item.category === 'Installation' ? 'bg-red-100 text-red-800' :
                        item.category === 'Model' ? 'bg-blue-100 text-blue-800' :
                        item.category === 'Performance' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-purple-100 text-purple-800'
                      }`}>
                        {item.category}
                      </span>
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 mb-2">{item.issue}</h3>
                      <p className="text-gray-600">{item.solution}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Additional Resources */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8 bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200"
          >
            <div className="flex items-center space-x-3 mb-6">
              <Lightbulb className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Additional Resources</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Documentation</h3>
                <div className="space-y-2">
                  <a
                    href="https://github.com/saminadamn/helipad-detection"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 transition-colors duration-200"
                  >
                    <ExternalLink className="h-4 w-4" />
                    <span>GitHub Repository</span>
                  </a>
                  <a
                    href="/technical"
                    className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 transition-colors duration-200"
                  >
                    <FileText className="h-4 w-4" />
                    <span>Technical Architecture</span>
                  </a>
                  <a
                    href="/results"
                    className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 transition-colors duration-200"
                  >
                    <Book className="h-4 w-4" />
                    <span>Results & Analysis</span>
                  </a>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Support</h3>
                <div className="space-y-2">
                  <p className="text-gray-600">
                    For questions, issues, or contributions, please visit our GitHub repository 
                    or contact the development team.
                  </p>
                  <div className="flex space-x-4">
                    <a
                      href="https://github.com/saminadamn/helipad-detection/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-primary"
                    >
                      Report Issue
                    </a>
                    <a
                      href="/demo"
                      className="btn-secondary"
                    >
                      Try Demo
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
  )
}

export default Documentation