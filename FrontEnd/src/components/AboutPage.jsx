import React from 'react';
import styles from './AboutPage.module.css'; // Impor CSS Module

const AboutPage = () => {
  return (
    <div className={styles.pageContent}>
      {/* BAGIAN BARU: Logo di bagian atas */}
      <div className={styles.logoContainer}>
        <img src="/Logo.png" alt="Logo Tanya Alkitab" className={styles.topLogo} />
      </div>

      {/* Header dengan teks yang sudah diubah */}
      <div className={styles.aboutHeader}>
        <p>
          "Tanya Alkitab" merupakan Chatbot yang dapat digunakan untuk menanyakan seputar pertanyan atau informasi Alkitab.
        </p>
      </div>

      {/* BAGIAN YANG TETAP SAMA: Tampilan kartu tim */}
      <div className={styles.teamContainer}>
        <div className={styles.teamCard}>
          <img src="/Vincent.png" alt="Vincent" className={styles.teamCardImg} />
          <div className={styles.teamCardInfo}>
            <h3>Vincent Calista</h3>
            <h3>535220075</h3>
            <p>Penulis</p>
          </div>
        </div>
        <div className={styles.teamCard}>
          <img src="/Viny.jpg" alt="Viny Christanti" className={styles.teamCardImg} />
          <div className={styles.teamCardInfo}>
            <h3>Viny Christanti Mawardi, S.Kom.,M.Kom.</h3>
            <p>Dosen Pembimbing</p>
          </div>
        </div>
        <div className={styles.teamCard}>
          <img src="/Manatap.jpg" alt="Manatap" className={styles.teamCardImg} />
          <div className={styles.teamCardInfo}>
            <h3>Manatap Dolok Lauro, S.Kom.,M.M.S.I.</h3>
            <p>Dosen Pendamping</p>
          </div>
        </div>
      </div>

      {/* BAGIAN BARU: Logo di bagian bawah */}
      <div className={styles.logoContainer}>
        <img src="/FTI_Untar.png" alt="Logo FTI Untar" className={styles.bottomLogo} />
      </div>
    </div>
  );
};

export default AboutPage;