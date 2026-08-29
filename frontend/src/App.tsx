import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './components/Sidebar'
import DashboardPage from './pages/DashboardPage'
import UploadPage from './pages/UploadPage'
import DocumentsPage from './pages/DocumentsPage'
import DocumentDetailPage from './pages/DocumentDetailPage'
import SearchPage from './pages/SearchPage'
import HealthPage from './pages/HealthPage'

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.22 } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.15 } },
}

export default function App() {
  const location = useLocation()

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <Routes location={location}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/documents/upload" element={<UploadPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/documents/:id" element={<DocumentDetailPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/health" element={<HealthPage />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
