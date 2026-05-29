import { motion } from 'framer-motion'

export default function StatsPanel({ stats, selectedLocation }) {
  return (
    <motion.div
      initial={{ x: 300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ delay: 0.3, duration: 0.8 }}
      className="absolute right-4 top-1/2 -translate-y-1/2 w-72 glass-panel rounded-2xl overflow-hidden z-10"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10">
        <h2 className="text-lg font-semibold text-neon-blue">Dashboard</h2>
        <p className="text-xs text-gray-400 mt-1">Real-time campus analytics</p>
      </div>

      {/* Stats Grid */}
      <div className="p-4 space-y-4">
        {/* Total Buildings */}
        <div className="glass-card rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400">Total Buildings</div>
              <div className="text-2xl font-bold text-neon-cyan">{stats.totalBuildings}</div>
            </div>
            <div className="text-3xl opacity-50">🏛️</div>
          </div>
        </div>

        {/* Total Area */}
        <div className="glass-card rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400">Campus Area</div>
              <div className="text-2xl font-bold text-neon-blue">{stats.totalArea}</div>
            </div>
            <div className="text-3xl opacity-50">📐</div>
          </div>
        </div>

        {/* Occupancy */}
        <div className="glass-card rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-xs text-gray-400">Occupancy Rate</div>
              <div className="text-2xl font-bold text-neon-purple">{Math.round(stats.occupancy)}%</div>
            </div>
            <div className="text-3xl opacity-50">👥</div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${stats.occupancy}%` }}
              transition={{ duration: 1 }}
              className="bg-gradient-to-r from-neon-purple to-neon-cyan h-2 rounded-full"
            />
          </div>
        </div>

        {/* Active Users */}
        <div className="glass-card rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400">Active Users</div>
              <div className="text-2xl font-bold text-green-400">{stats.activeUsers}</div>
            </div>
            <div className="text-3xl opacity-50">📱</div>
          </div>
        </div>
      </div>

      {/* Selected Location Details */}
      {selectedLocation && (
        <div className="px-4 py-3 border-t border-white/10 bg-neon-cyan/5">
          <div className="text-xs text-gray-400 mb-2">Selected Location</div>
          <div className="text-sm font-semibold text-neon-cyan">{selectedLocation.name}</div>
          <div className="text-xs text-gray-400 mt-1">
            Type: {selectedLocation.type}
          </div>
          <div className="text-xs text-gray-400">
            Coordinates: {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/10 bg-black/20">
        <div className="text-xs text-gray-400 text-center">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </motion.div>
  )
}
