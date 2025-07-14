import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Plane, 
  Target, 
  Zap, 
  Shield, 
  Brain, 
  BarChart3,
  ArrowRight,
  CheckCircle,
  Clock,
  Eye,
  Cpu
} from 'lucide-react'

const Home = () => {
  const features = [
    {
      icon: Target,
      title: 'High Accuracy',
      description: '92.5% test accuracy on real helipad images',
      color: 'text-green-600'
    },
    {
      icon: Zap,
      title: 'Fast Inference',
      description: '<100ms prediction time per image',
      color: 'text-yellow-600'
    },
    {
      icon: Brain,
      title: 'Deep Learning',
      description: 'CNN-based architecture with advanced features',
      color: 'text-purple-600'
    },
    {
      icon: Shield,
      title: 'Robust Performance',
      description: 'Handles various lighting and weather conditions',
      color: 'text-blue-600'
    }
  ]

  const applications = [
    {
      icon: Plane,
      title: 'Aviation Safety',
      description: 'Automated helipad identification for flight planning and emergency response'
    },
    {
      icon: Eye,
      title: 'Autonomous Navigation',
      description: 'Landing site detection for autonomous aircraft and drone systems'
    },
    {
      icon: Shield,
      title: 'Emergency Services',
      description: 'Rapid landing site assessment during disasters and medical evacuations'
    }
  ]

  const stats = [
    { label: 'Test Accuracy', value: '92.5%', icon: Target },
    { label: 'Training Images', value: '200+', icon: BarChart3 },
    { label: 'Inference Time', value: '<100ms', icon: Clock },
    { label: 'Model Size', value: 'Optimized', icon: Cpu }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mb-8"
            >
              <div className="inline-flex items-center justify-center p-4 bg-white bg-opacity-20 rounded-full mb-6">
                <Plane className="h-12 w-12 text-white animate-float" />
              </div>
              <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
                AI-Powered Helipad
                <span className="block text-primary-200">Detection System</span>
              </h1>
              <p className="text-xl md:text-2xl text-primary-100 mb-8 max-w-3xl mx-auto">
                Automated helipad identification in aerial imagery using Convolutional Neural Networks 
                with <span className="font-semibold text-white">92.5% accuracy</span>
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex flex-col sm:flex-row gap-4 justify-center"
            >
              <Link to="/demo" className="btn-primary bg-white text-primary-600 hover:bg-gray-100">
                Try Live Demo
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
              <Link to="/technical" className="btn-secondary border-white text-white hover:bg-white hover:text-primary-600">
                Technical Details
              </Link>
            </motion.div>
          </div>
        </div>

        {/* Floating elements */}
        <div className="absolute top-20 left-10 opacity-20">
          <div className="w-20 h-20 bg-white rounded-full animate-pulse-slow"></div>
        </div>
        <div className="absolute bottom-20 right-10 opacity-20">
          <div className="w-16 h-16 bg-white rounded-full animate-bounce-slow"></div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mb-4">
                  <stat.icon className="h-6 w-6 text-primary-600" />
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Key Features
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Advanced deep learning capabilities designed for real-world aviation applications
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card p-6 text-center hover:scale-105 transition-transform duration-300"
              >
                <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-4`}>
                  <feature.icon className={`h-8 w-8 ${feature.color}`} />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Applications Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Real-World Applications
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Enabling safer and more efficient aviation operations across multiple industries
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {applications.map((app, index) => (
              <motion.div
                key={app.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                className="card p-8 text-center"
              >
                <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-6">
                  <app.icon className="h-8 w-8 text-primary-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">{app.title}</h3>
                <p className="text-gray-600">{app.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Highlights */}
      <section className="py-20 bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                Advanced CNN Architecture
              </h2>
              <p className="text-xl text-gray-300 mb-8">
                Built with TensorFlow/Keras using state-of-the-art deep learning techniques 
                for robust image classification and real-time inference.
              </p>
              
              <div className="space-y-4">
                {[
                  'Progressive Feature Extraction (32→64→128 filters)',
                  'Regularization with Dropout layers',
                  'Optimal 3-layer convolutional architecture',
                  'Real dataset training with 200+ helipad images'
                ].map((item, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0" />
                    <span className="text-gray-300">{item}</span>
                  </div>
                ))}
              </div>

              <div className="mt-8">
                <Link to="/technical" className="btn-primary">
                  Explore Architecture
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-gray-800 rounded-xl p-6"
            >
              <h3 className="text-lg font-semibold mb-4 text-primary-400">Model Performance</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Test Accuracy</span>
                  <span className="text-2xl font-bold text-green-400">92.5%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div className="bg-green-400 h-2 rounded-full" style={{ width: '92.5%' }}></div>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Training Accuracy</span>
                  <span className="text-2xl font-bold text-blue-400">95.2%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div className="bg-blue-400 h-2 rounded-full" style={{ width: '95.2%' }}></div>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Validation Accuracy</span>
                  <span className="text-2xl font-bold text-purple-400">91.8%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div className="bg-purple-400 h-2 rounded-full" style={{ width: '91.8%' }}></div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Ready to Experience AI-Powered Detection?
            </h2>
            <p className="text-xl text-primary-100 mb-8 max-w-2xl mx-auto">
              Try our interactive demo to see the helipad detection system in action, 
              or explore the technical implementation details.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/demo" className="btn-primary bg-white text-primary-600 hover:bg-gray-100">
                Try Interactive Demo
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
              <Link to="/documentation" className="btn-secondary border-white text-white hover:bg-white hover:text-primary-600">
                View Documentation
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default Home