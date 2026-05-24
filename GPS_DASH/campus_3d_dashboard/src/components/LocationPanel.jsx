import { motion } from 'framer-motion'

export default function LocationPanel({ locations, selectedLocation, onSelectLocation, onPathFind }) {
  const getIconForType = (type) => {
    switch (type) {
      case 'building':
        return '🏢'
      case 'facility':
        return '🏪'
      case 'entrance':
        return '🚪'
      default:
        return '📍'
    }
  }

  const getColorForType = (type) => {
    switch (type) {
      case 'building':
        return 'border-neon-cyan/50 bg-neon-cyan/10'
      case 'facility':
        return 'border-neon-blue/50 bg-neon-blue/10'
      case 'entrance':
        return 'border-neon-purple/50 bg-neon-purple/10'
      default:
        return 'border-gray/50 bg-gray/10'
    }
  }

  return (
    <motion.div
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ delay: 0.3, duration: 0.8 }}
      className="absolute left-4 top-1/2 -translate-y-1/2 w-72 glass-panel rounded-2xl overflow-hidden z-10"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10">
        <h2 className="text-lg font-semibold text-neon-cyan">Locations</h2>
        <p className="text-xs text-gray-400 mt-1">Click to select • Right-click for path</p>
      </div>

      {/* Location List */}
      <div className="max-h-96 overflow-y-auto p-2 space-y-1">
        {locations.map((location, index) => (
          <motion.div
            key={location.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`
              p-3 rounded-lg cursor-pointer transition-all duration-200
              border ${selectedLocation?.id === location.id 
                ? `${getColorForType(location.type)} border-2` 
                : 'border-white/5 hover:border-white/20 hover:bg-white/5'}
            `}
            onClick={() => onSelectLocation(location)}
            onContextMenu={(e) => {
              e.preventDefault()
              if (selectedLocation) {
                onPathFind(selectedLocation, location)
              }
            }}
          >
            <div className="flex items-center gap-3">
              <span className="text-xl">{getIconForType(location.type)}</span>
              <div className="flex-1">
                <div className="text-sm font-medium text-white">{location.name}</div>
                <div className="text-xs text-gray-400">
                  {location.lat.toFixed(4)}, {location.lng.toFixed(4)}
                </div>
              </div>
              {selectedLocation?.id === location.id && (
                <div className="w-2 h-2 bg-neon-cyan rounded-full animate-pulse"></div>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/10 bg-black/20">
        <div className="text-xs text-gray-400 text-center">
          {locations.length} locations available
        </div>
      </div>
    </motion.div>
  )
}
