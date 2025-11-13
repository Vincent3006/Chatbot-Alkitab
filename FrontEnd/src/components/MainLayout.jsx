import React, { useState, useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import { FaBars } from 'react-icons/fa';
import styles from './MainLayout.module.css'; // Impor CSS Module

const MainLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);
  const location = useLocation();

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  useEffect(() => {
    if (window.innerWidth <= 768) {
      setSidebarOpen(false);
    }
  }, [location]);

  useEffect(() => {
    const handleResize = () => {
      setSidebarOpen(window.innerWidth > 768);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Menggunakan template literal untuk menggabungkan class dari CSS Module
  const mainContentClass = `${styles.mainContent} ${sidebarOpen ? styles.sidebarPushed : ''}`;

  return (
    <div className={styles.appLayout}>
      <Sidebar isOpen={sidebarOpen} toggleSidebar={toggleSidebar} />
      <div className={mainContentClass}>
        <nav className={styles.navbar}>
          <button onClick={toggleSidebar} className={styles.menuButton}>
            <FaBars />
          </button>
          <div className={styles.navbarLogoContainer}>
            <img src="/logo_panjang.jpeg" alt="Logo Tanya Alkitab" className={styles.navbarLogo}/>
          </div>
        </nav>
        <main className={styles.contentArea}>
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;