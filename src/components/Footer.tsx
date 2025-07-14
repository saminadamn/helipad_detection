import { Github, Linkedin, Mail, Plane } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Plane className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold">Helipad Detection System</span>
            </div>
            <p className="text-gray-400 mb-4 max-w-md">
              AI-powered helipad detection in aerial imagery using Convolutional Neural Networks. 
              Achieving 92.5% accuracy for aviation safety and autonomous navigation applications.
            </p>
            <div className="flex space-x-4">
              <a
                href="https://github.com/saminadamn/helipad-detection"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Github className="h-6 w-6" />
              </a>
              <a
                href="https://linkedin.com/in/samina-parveen"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Linkedin className="h-6 w-6" />
              </a>
              <a
                href="mailto:samina@example.com"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Mail className="h-6 w-6" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <a href="/demo" className="text-gray-400 hover:text-white transition-colors duration-200">
                  Try Demo
                </a>
              </li>
              <li>
                <a href="/technical" className="text-gray-400 hover:text-white transition-colors duration-200">
                  Technical Details
                </a>
              </li>
              <li>
                <a href="/documentation" className="text-gray-400 hover:text-white transition-colors duration-200">
                  Documentation
                </a>
              </li>
              <li>
                <a href="/results" className="text-gray-400 hover:text-white transition-colors duration-200">
                  Results & Analysis
                </a>
              </li>
            </ul>
          </div>

          {/* Technical Info */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Technical</h3>
            <ul className="space-y-2 text-gray-400">
              <li>TensorFlow/Keras</li>
              <li>Convolutional Neural Networks</li>
              <li>Computer Vision</li>
              <li>Deep Learning</li>
              <li>Image Classification</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
          <p>&copy; 2025 Helipad Detection System. Built with React, TypeScript, and Tailwind CSS.</p>
          <p className="mt-2">
            Developed by{' '}
            <a
              href="https://github.com/saminadamn"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-400 hover:text-primary-300 transition-colors duration-200"
            >
              Samina Parveen
            </a>
          </p>
        </div>
      </div>
    </footer>
  )
}

export default Footer