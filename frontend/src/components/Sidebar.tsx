import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  FileText,
  Search,
  Heart,
  Cpu,
} from 'lucide-react'

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/documents/upload', icon: Upload, label: 'Upload PDF' },
  { to: '/documents', icon: FileText, label: 'Documents' },
  { to: '/search', icon: Search, label: 'AI Search' },
  { to: '/health', icon: Heart, label: 'System Health' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-text">
          OMNI<span>BRAIN</span>
        </div>
        <div className="sidebar-logo-sub">Agentic RAG System</div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="sidebar-nav-section">
          <div className="sidebar-section-label">Navigation</div>
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon />
              {label}
            </NavLink>
          ))}
        </div>

        <div className="sidebar-nav-section">
          <div className="sidebar-section-label">System</div>
          <div className="nav-item" style={{ cursor: 'default', opacity: .5 }}>
            <Cpu />
            Multi-Modal RAG
          </div>
        </div>
      </nav>

      {/* Bottom badge */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid rgba(255,255,255,.08)' }}>
        <div style={{ fontSize: 11, color: 'rgba(255,255,255,.3)', fontWeight: 500 }}>
          v1.0.0 · gobinath branch
        </div>
      </div>
    </aside>
  )
}
