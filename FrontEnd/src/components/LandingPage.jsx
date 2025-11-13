import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './LandingPage.module.css'; 
const LandingPage = () => {
  const navigate = useNavigate();
  const handleStart = () => {
    navigate('/app');
  };
  return (
    <div className={styles.landingContainer}>
      <div className={styles.landingContent}>
        <img src="/logo_square.jpeg" alt="Tanya Alkitab Log" className={styles.landingLogo} />
        <h1 className={styles.landingTitle}>Selamat Datang</h1>
        <p className={styles.landingDescription}>
          Anda dapat menanyakan informasi seputar Alkitab di "Tanya Alkitab"
        </p>
        <button onClick={handleStart} className={styles.landingButton}>
          Mulai Tanya Alkitab
        </button>
      </div>
    </div>
  );
};
export default LandingPage;