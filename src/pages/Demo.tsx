import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  Upload, 
  Image as ImageIcon, 
  Target, 
  Clock, 
  BarChart3,
  CheckCircle,
  AlertCircle,
  Loader,
  Download,
  RotateCcw
} from 'lucide-react'

interface PredictionResult {
  prediction: 'Helipad' | 'No Helipad'
  confidence: number
  processingTime: number
  probabilities: {
    helipad: number
    noHelipad: number
  }
}

const Demo = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [fileName, setFileName] = useState<string>('')

  // Sample images for testing
  const sampleImages = [
    {
      name: 'Hospital Helipad',
      url: 'https://images.pexels.com/photos/8460157/pexels-photo-8460157.jpeg?auto=compress&cs=tinysrgb&w=400',
      expectedResult: 'Helipad'
    },
    {
      name: 'Urban Rooftop',
      url: 'https://images.pexels.com/photos/2462015/pexels-photo-2462015.jpeg?auto=compress&cs=tinysrgb&w=400',
      expectedResult: 'No Helipad'
    },
    {
      name: 'Airport View',
      url: 'https://images.pexels.com/photos/2026324/pexels-photo-2026324.jpeg?auto=compress&cs=tinysrgb&w=400',
      expectedResult: 'Helipad'
    }
  ]

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = () => {
        setSelectedImage(reader.result as string)
        setResult(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp']
    },
    multiple: false
  })

  const simulateProcessing = async () => {
    setIsProcessing(true)
    setResult(null)

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000))

    // Simulate AI prediction (in real implementation, this would call your ML model)
    const isHelipad = Math.random() > 0.3 // 70% chance of detecting helipad for demo
    const confidence = 0.85 + Math.random() * 0.1 // Random confidence between 85-95%
    
    const mockResult: PredictionResult = {
      prediction: isHelipad ? 'Helipad' : 'No Helipad',
      confidence: confidence,
      processingTime: 45 + Math.random() * 30, // Random time between 45-75ms
      probabilities: {
        helipad: isHelipad ? confidence : 1 - confidence,
        noHelipad: isHelipad ? 1 - confidence : confidence
      }
    }

    setResult(mockResult)
    setIsProcessing(false)
  }

  const loadSampleImage = (imageUrl: string, name: string) => {
    setSelectedImage(imageUrl)
    setFileName(name)
    setResult(null)
  }

  const resetDemo = () => {
    setSelectedImage(null)
    setResult(null)
    setFileName('')
    setIsProcessing(false)
  }

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
              Interactive Demo
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Upload an aerial image or select a sample to see our AI-powered helipad detection system in action. 
              Experience real-time analysis with confidence scoring.
            </p>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-6"
          >
            <div className="card p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">Upload Image</h2>
              
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
                  isDragActive
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                {isDragActive ? (
                  <p className="text-primary-600 font-medium">Drop the image here...</p>
                ) : (
                  <div>
                    <p className="text-gray-600 mb-2">
                      Drag & drop an aerial image here, or click to select
                    </p>
                    <p className="text-sm text-gray-500">
                      Supports JPG, PNG, BMP formats
                    </p>
                  </div>
                )}
              </div>

              {/* Sample Images */}
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Or try sample images:</h3>
                <div className="grid grid-cols-3 gap-3">
                  {sampleImages.map((sample, index) => (
                    <button
                      key={index}
                      onClick={() => loadSampleImage(sample.url, sample.name)}
                      className="relative group overflow-hidden rounded-lg border-2 border-gray-200 hover:border-primary-400 transition-colors duration-200"
                    >
                      <img
                        src={sample.url}
                        alt={sample.name}
                        className="w-full h-20 object-cover group-hover:scale-105 transition-transform duration-200"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-opacity duration-200"></div>
                      <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white text-xs p-1 text-center">
                        {sample.name}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Selected Image Preview */}
            {selectedImage && (
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Selected Image</h3>
                  <button
                    onClick={resetDemo}
                    className="text-gray-500 hover:text-gray-700 transition-colors duration-200"
                  >
                    <RotateCcw className="h-5 w-5" />
                  </button>
                </div>
                <div className="relative">
                  <img
                    src={selectedImage}
                    alt="Selected for analysis"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                  {fileName && (
                    <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white text-sm px-2 py-1 rounded">
                      {fileName}
                    </div>
                  )}
                </div>
                
                <button
                  onClick={simulateProcessing}
                  disabled={isProcessing}
                  className="w-full mt-4 btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <>
                      <Loader className="animate-spin h-5 w-5 mr-2" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Target className="h-5 w-5 mr-2" />
                      Analyze Image
                    </>
                  )}
                </button>
              </div>
            )}
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-6"
          >
            {/* Processing Status */}
            {isProcessing && (
              <div className="card p-6">
                <div className="flex items-center justify-center space-x-3">
                  <Loader className="animate-spin h-6 w-6 text-primary-600" />
                  <span className="text-lg font-medium text-gray-900">
                    Analyzing image with AI model...
                  </span>
                </div>
                <div className="mt-4 bg-gray-200 rounded-full h-2">
                  <div className="bg-primary-600 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
                </div>
              </div>
            )}

            {/* Results */}
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="card p-6"
              >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">Analysis Results</h3>
                
                {/* Main Result */}
                <div className={`p-4 rounded-lg mb-6 ${
                  result.prediction === 'Helipad' 
                    ? 'bg-green-50 border border-green-200' 
                    : 'bg-red-50 border border-red-200'
                }`}>
                  <div className="flex items-center space-x-3">
                    {result.prediction === 'Helipad' ? (
                      <CheckCircle className="h-8 w-8 text-green-600" />
                    ) : (
                      <AlertCircle className="h-8 w-8 text-red-600" />
                    )}
                    <div>
                      <div className={`text-2xl font-bold ${
                        result.prediction === 'Helipad' ? 'text-green-800' : 'text-red-800'
                      }`}>
                        {result.prediction}
                      </div>
                      <div className="text-sm text-gray-600">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Detailed Metrics */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Clock className="h-5 w-5 text-gray-600" />
                      <span className="font-medium text-gray-900">Processing Time</span>
                    </div>
                    <div className="text-2xl font-bold text-primary-600">
                      {result.processingTime.toFixed(0)}ms
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <BarChart3 className="h-5 w-5 text-gray-600" />
                      <span className="font-medium text-gray-900">Accuracy</span>
                    </div>
                    <div className="text-2xl font-bold text-green-600">92.5%</div>
                  </div>
                </div>

                {/* Probability Breakdown */}
                <div className="space-y-3">
                  <h4 className="font-medium text-gray-900">Probability Breakdown:</h4>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Helipad</span>
                      <span className="text-sm font-medium">
                        {(result.probabilities.helipad * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                        style={{ width: `${result.probabilities.helipad * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">No Helipad</span>
                      <span className="text-sm font-medium">
                        {(result.probabilities.noHelipad * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-red-500 h-2 rounded-full transition-all duration-1000"
                        style={{ width: `${result.probabilities.noHelipad * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-3 mt-6">
                  <button
                    onClick={() => simulateProcessing()}
                    className="btn-secondary flex-1"
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reanalyze
                  </button>
                  <button className="btn-primary flex-1">
                    <Download className="h-4 w-4 mr-2" />
                    Export Results
                  </button>
                </div>
              </motion.div>
            )}

            {/* Instructions */}
            {!selectedImage && !isProcessing && !result && (
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">How to Use</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-6 h-6 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      1
                    </div>
                    <p className="text-gray-600">
                      Upload an aerial image or select from sample images
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-6 h-6 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      2
                    </div>
                    <p className="text-gray-600">
                      Click "Analyze Image" to run the AI detection
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-6 h-6 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      3
                    </div>
                    <p className="text-gray-600">
                      View detailed results with confidence scores and processing time
                    </p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* Technical Note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-12 card p-6 bg-blue-50 border-blue-200"
        >
          <div className="flex items-start space-x-3">
            <ImageIcon className="h-6 w-6 text-blue-600 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-blue-900 mb-2">Demo Note</h3>
              <p className="text-blue-800">
                This is a simulated demo for demonstration purposes. In the actual implementation, 
                the system uses a trained TensorFlow/Keras CNN model with 92.5% accuracy on real helipad images. 
                The model processes 224x224 pixel images and provides real-time inference in under 100ms.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Demo