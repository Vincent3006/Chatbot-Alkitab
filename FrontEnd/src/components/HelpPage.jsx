import React from 'react';
import styles from './HelpPage.module.css'; // Impor CSS Module

const HelpPage = () => {
  return (
    <div className={styles.pageContent}>
      <h2>Pusat Bantuan</h2>
      <p>
        Selamat datang di pusat bantuan Tanya Alkitab. Berikut adalah beberapa pertanyaan yang sering diajukan.
      </p>
      
      <div className={styles.faqItem}>
        <h3>Bagaimana Cara Bertanya?</h3>
        <p>
          Pada halaman "Chat", Anda akan menemukan kolom input di bagian bawah. Cukup ketik pertanyaan Anda terkait Alkitab dan tekan tombol kirim atau tombol Enter. Tanya Alkitab akan memproses dan memberikan jawaban untuk Anda.
        </p>
      </div>
      
      <div className={styles.faqItem}>
        <h3>Apakah Jawaban Selalu Akurat?</h3>
        <p>
          Tanya Alkitab dirancang untuk memberikan jawaban berdasarkan konteks Alkitab. Namun, Tanya Alkitab adalah alat bantu. Kami sangat menyarankan Anda untuk tetap mengecek terlebih dahulu atau dapat menanyakan secara langsung terhadap mentor rohani Anda.
        </p>
      </div>

      <div className={styles.faqItem}>
        <h3>Kontak</h3>
        <p>
          Jika Anda memiliki pertanyaan lebih lanjut atau menemukan masalah, jangan ragu untuk menghubungi :.
        </p>
          <p>
          Nama          : Vincent Calista
        </p>
          <p>
          Nomor Telpon  : (+62) 858-8088-5760
        </p>
          <p>
          Email         : vincent.535220075@stu.untar.ac.id
        </p>
      </div>
    </div>
  );
};

export default HelpPage;