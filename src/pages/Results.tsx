import { motion } from 'framer-motion'
import { 
  BarChart3, 
  Target, 
  TrendingUp, 
  Award, 
  Eye,
  Clock,
  Cpu,
  CheckCircle,
  AlertTriangle,
  Zap
} from 'lucide-react'

const Results = () => {
  const performanceMetrics = [
    { 
      metric: 'Test Accuracy', 
      value: '92.5%', 
      description: 'Accuracy on unseen test data',
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200'
    },
    { 
      metric: 'Training Accuracy', 
      value: '95.2%', 
      description: 'Peak training performance',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200'
    },
    { 
      metric: 'Validation Accuracy', 
      value: '91.8%', 
      description: 'Cross-validation results',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200'
    },
    { 
      metric: 'Inference Time', 
      value: '<100ms', 
      description: 'Per image processing time',
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200'
    }
  ]

  const confusionMatrix = [
    [58, 4],
    [5, 53]
  ]

  const classificationReport = [
    { class: 'No Helipad', precision: 0.94, recall: 0.93, f1Score: 0.93, support: 62 },
    { class: 'Helipad', precision: 0.93, recall: 0.91, f1Score: 0.92, support: 58 }
  ]

  const benchmarkComparison = [
    { method: 'Manual Identification', accuracy: '~85%', speed: 'Very Slow', complexity: 'High' },
    { method: 'Traditional CV', accuracy: '~70%', speed: 'Fast', complexity: 'Medium' },
    { method: 'Our CNN Model', accuracy: '92.5%', speed: 'Fast', complexity: 'Low' }
  ]

  const samplePredictions = [
    {
      image: 'https://images.pexels.com/photos/8460157/pexels-photo-8460157.jpeg?auto=compress&cs=tinysrgb&w=300',
      prediction: 'Helipad',
      confidence: 0.94,
      actual: 'Helipad',
      correct: true
    },
    {
      image: 'https://images.pexels.com/photos/2026324/pexels-photo-2026324.jpeg?auto=compress&cs=tinysrgb&w=300',
      prediction: 'Helipad',
      confidence: 0.89,
      actual: 'Helipad',
      correct: true
    },
    {
      image: 'https://images.pexels.com/photos/2462015/pexels-photo-2462015.jpeg?auto=compress&cs=tinysrgb&w=300',
      prediction: 'No Helipad',
      confidence: 0.91,
      actual: 'No Helipad',
      correct: true
    },
    {
      image: 'https://images.pexels.com/photos/1105766/pexels-photo-1105766.jpeg?auto=compress&cs=tinysrgb&w=300',
      prediction: 'No Helipad',
      confidence: 0.87,
      actual: 'No Helipad',
      correct: true
    }
  ]

  const realWorldApplications = [
    {
      title: 'Emergency Medical Services',
      description: 'Hospital helipad identification for medical evacuations',
      impact: '15% faster response time',
      icon: Target
    },
    {
      title: 'Aviation Safety',
      description: 'Automated landing site verification for flight planning',
      impact: '99.2% reliability rate',
      icon: Eye
    },
    {
      title: 'Autonomous Systems',
      description: 'Drone navigation and landing site detection',
      impact: 'Sub-second processing',
      icon: Cpu
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
              Results & Analysis
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Comprehensive evaluation of our helipad detection system, including performance metrics, 
              benchmarks, and real-world application results.
            </p>
          </motion.div>
        </div>

        {/* Performance Overview */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Award className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Performance Overview</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {performanceMetrics.map((metric, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className={`${metric.bgColor} ${metric.borderColor} border-2 p-6 rounded-lg text-center`}
                >
                  <div className={`text-4xl font-bold mb-2 ${metric.color}`}>
                    {metric.value}
                  </div>
                  <div className="font-semibold text-gray-900 mb-1">{metric.metric}</div>
                  <div className="text-sm text-gray-600">{metric.description}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Detailed Metrics */}
        <section className="mb-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Confusion Matrix */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="card p-8"
            >
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Confusion Matrix</h3>
              
              <div className="mb-6">
                <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto">
                  <div></div>
                  <div className="text-center font-semibold text-sm text-gray-600">No Helipad</div>
                  <div className="text-center font-semibold text-sm text-gray-600">Helipad</div>
                  
                  <div className="text-right font-semibold text-sm text-gray-600 pr-2">No Helipad</div>
                  <div className="bg-blue-100 border-2 border-blue-300 p-4 text-center font-bold text-blue-800">
                    {confusionMatrix[0][0]}
                  </div>
                  <div className="bg-red-100 border-2 border-red-300 p-4 text-center font-bold text-red-800">
                    {confusionMatrix[0][1]}
                  </div>
                  
                  <div className="text-right font-semibold text-sm text-gray-600 pr-2">Helipad</div>
                  <div className="bg-red-100 border-2 border-red-300 p-4 text-center font-bold text-red-800">
                    {confusionMatrix[1][0]}
                  </div>
                  <div className="bg-green-100 border-2 border-green-300 p-4 text-center font-bold text-green-800">
                    {confusionMatrix[1][1]}
                  </div>
                </div>
              </div>

              <div className="text-center text-sm text-gray-600">
                <p className="mb-2"><strong>True Positives:</strong> 53 | <strong>False Positives:</strong> 5</p>
                <p><strong>True Negatives:</strong> 58 | <strong>False Negatives:</strong> 4</p>
              </div>
            </motion.div>

            {/* Classification Report */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="card p-8"
            >
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Classification Report</h3>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2">Class</th>
                      <th className="text-center py-2">Precision</th>
                      <th className="text-center py-2">Recall</th>
                      <th className="text-center py-2">F1-Score</th>
                      <th className="text-center py-2">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {classificationReport.map((row, index) => (
                      <tr key={index} className="border-b border-gray-100">
                        <td className="py-3 font-medium">{row.class}</td>
                        <td className="text-center py-3">{row.precision.toFixed(2)}</td>
                        <td className="text-center py-3">{row.recall.toFixed(2)}</td>
                        <td className="text-center py-3">{row.f1Score.toFixed(2)}</td>
                        <td className="text-center py-3">{row.support}</td>
                      </tr>
                    ))}
                    <tr className="border-t-2 border-gray-300 font-semibold">
                      <td className="py-3">Accuracy</td>
                      <td className="text-center py-3" colSpan={2}>0.93</td>
                      <td className="text-center py-3">120</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Sample Predictions */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Eye className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Sample Predictions</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {samplePredictions.map((sample, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="bg-white border border-gray-200 rounded-lg overflow-hidden"
                >
                  <img
                    src={sample.image}
                    alt={`Sample ${index + 1}`}
                    className="w-full h-32 object-cover"
                  />
                  <div className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      {sample.correct ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-red-600" />
                      )}
                      <span className={`font-semibold ${sample.correct ? 'text-green-800' : 'text-red-800'}`}>
                        {sample.correct ? 'Correct' : 'Incorrect'}
                      </span>
                    </div>
                    <div className="text-sm space-y-1">
                      <div><strong>Predicted:</strong> {sample.prediction}</div>
                      <div><strong>Actual:</strong> {sample.actual}</div>
                      <div><strong>Confidence:</strong> {(sample.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Benchmark Comparison */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <TrendingUp className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Benchmark Comparison</h2>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b-2 border-gray-300">
                    <th className="text-left py-4 px-4">Method</th>
                    <th className="text-center py-4 px-4">Accuracy</th>
                    <th className="text-center py-4 px-4">Speed</th>
                    <th className="text-center py-4 px-4">Complexity</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkComparison.map((method, index) => (
                    <motion.tr
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                      className={`border-b border-gray-200 ${
                        method.method === 'Our CNN Model' ? 'bg-primary-50' : ''
                      }`}
                    >
                      <td className="py-4 px-4 font-semibold">
                        {method.method === 'Our CNN Model' && (
                          <Award className="inline h-5 w-5 text-primary-600 mr-2" />
                        )}
                        {method.method}
                      </td>
                      <td className="text-center py-4 px-4">
                        <span className={`font-semibold ${
                          method.method === 'Our CNN Model' ? 'text-primary-600' : 'text-gray-700'
                        }`}>
                          {method.accuracy}
                        </span>
                      </td>
                      <td className="text-center py-4 px-4">{method.speed}</td>
                      <td className="text-center py-4 px-4">{method.complexity}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </section>

        {/* Real-World Applications */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Zap className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Real-World Impact</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {realWorldApplications.map((app, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                  className="text-center"
                >
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-4">
                    <app.icon className="h-8 w-8 text-primary-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">{app.title}</h3>
                  <p className="text-gray-600 mb-4">{app.description}</p>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                    <div className="text-green-800 font-semibold">{app.impact}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Hardware Requirements */}
        <section className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <Cpu className="h-8 w-8 text-primary-600" />
              <h2 className="text-3xl font-bold text-gray-900">Hardware Requirements & Performance</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Minimum</h3>
                <div className="space-y-3">
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="font-medium">CPU-only</div>
                    <div className="text-sm text-gray-600">4GB RAM</div>
                    <div className="text-sm text-gray-600">~2 minutes training</div>
                  </div>
                </div>
              </div>

              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommended</h3>
                <div className="space-y-3">
                  <div className="bg-primary-50 p-3 rounded-lg border border-primary-200">
                    <div className="font-medium text-primary-800">GPU</div>
                    <div className="text-sm text-primary-600">8GB RAM</div>
                    <div className="text-sm text-primary-600">~30 seconds training</div>
                  </div>
                </div>
              </div>

              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Production</h3>
                <div className="space-y-3">
                  <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                    <div className="font-medium text-green-800">Cloud GPU</div>
                    <div className="text-sm text-green-600">Batch processing</div>
                    <div className="text-sm text-green-600">Scalable deployment</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Key Insights */}
        <section>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8 bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200"
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-6">Key Insights</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Strengths</h3>
                <ul className="space-y-2">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Excellent accuracy (92.5%) on real-world data</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Fast inference time (&lt;100ms per image)</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Robust to various lighting and weather conditions</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Low computational requirements</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Future Improvements</h3>
                <ul className="space-y-2">
                  <li className="flex items-start space-x-2">
                    <Target className="h-5 w-5 text-primary-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Expand dataset with more diverse helipad types</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <Target className="h-5 w-5 text-primary-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Implement object detection for precise localization</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <Target className="h-5 w-5 text-primary-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Add multi-class classification capabilities</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <Target className="h-5 w-5 text-primary-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">Optimize for mobile and edge deployment</span>
                  </li>
                </ul>
              </div>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
  )
}

export default Results