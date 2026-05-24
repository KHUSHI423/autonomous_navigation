import { motion } from 'framer-motion'

export default function HUDOverlay({ time, selectedLocation }) {
  return (
    <motion.div
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="absolute top-0 left-0 right-0 glass-panel px-6 py-4 flex justify-between items-center z-10"
    >
      {/* Left Section - Logo & Title */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-cyan to-neon-blue flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold neon-text text-neon-cyan">X to Y Path Simulation</h1>
            <p className="text-xs text-gray-400">Real-time 3D Navigation System</p>
          </div>
        </div>
      </div>

      {/* Center Section - Time & Date */}
      <div className="flex flex-col items-center">
        <div className="text-2xl font-mono font-bold text-neon-blue neon-text">
          {time.toLocaleTimeString('en-US', { hour12: false })}
        </div>
        <div className="text-xs text-gray-400">
          {time.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
        </div>
      </div>

      {/* Right Section - Quick Stats */}
      <div className="flex items-center gap-6">
        <div className="text-right">
          <div className="text-xs text-gray-400">System Status</div>
          <div className="flex items-center gap-2 justify-end">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400 font-semibold">Online</span>
          </div>
        </div>
        
        <div className="h-8 w-px bg-gray-700"></div>
        
        {selectedLocation && (
          <div className="text-right">
            <div className="text-xs text-gray-400">Selected</div>
            <div className="text-sm text-neon-cyan font-semibold">{selectedLocation.name}</div>
          </div>
        )}
      </div>
    </motion.div>
  )
}
