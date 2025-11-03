import React from 'react';
import { NavLink } from 'react-router-dom';
// Tambahkan FaComment untuk ikon chat
import { FaInfoCircle, FaQuestionCircle, FaTimes, FaComment } from 'react-icons/fa';
import styles from './Sidebar.module.css'; // Impor CSS Module

const Sidebar = ({ isOpen, toggleSidebar }) => {
  return (
    <>
      <div 
        className={`${styles.sidebarOverlay} ${isOpen ? styles.show : ''}`} 
        onClick={toggleSidebar}
      ></div>
      <aside className={`${styles.sidebar} ${isOpen ? styles.open : ''}`}>
        <div className={styles.sidebarHeader}>
          <h3>Menu</h3>
          <button onClick={toggleSidebar} className={styles.sidebarCloseBtn}>
            <FaTimes />
          </button>
        </div>
        <nav className={styles.sidebarNav}>
          {/* Tombol Chat Baru */}
          <NavLink to="/app/chat" className={({ isActive }) => isActive ? `${styles.sidebarLink} ${styles.active}` : styles.sidebarLink} onClick={toggleSidebar}>
            <FaComment className={styles.sidebarIcon} />
            <span>Chat</span>
          </NavLink>
          <NavLink to="/app/about" className={({ isActive }) => isActive ? `${styles.sidebarLink} ${styles.active}` : styles.sidebarLink} onClick={toggleSidebar}>
            <FaInfoCircle className={styles.sidebarIcon} />
            <span>About</span>
          </NavLink>
          <NavLink to="/app/help" className={({ isActive }) => isActive ? `${styles.sidebarLink} ${styles.active}` : styles.sidebarLink} onClick={toggleSidebar}>
            <FaQuestionCircle className={styles.sidebarIcon} />
            <span>Help</span>
          </NavLink>
        </nav>
      </aside>
    </>
  );
};

export default Sidebar;