import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Current Upload", end: true },
  { to: "/results", label: "Prediction Results" },
  { to: "/history", label: "Archived History" },
  { to: "/health", label: "System Health" },
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-logo">✈</span>
        <span>Maintenance NLP</span>
      </div>
      <nav className="sidebar-nav">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.end}
            className={({ isActive }) =>
              "sidebar-link" + (isActive ? " active" : "")
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-foot">v0.1 · mock backend</div>
    </aside>
  );
}

export default Sidebar;
