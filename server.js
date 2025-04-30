require('dotenv').config();
const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Conexión a Neon.tech
const pool = new Pool({
  connectionString: process.env.NEON_DATABASE_URL,
  ssl: {
    rejectUnauthorized: false
  }
});

// Ruta de prueba de conexión
app.get('/api/test', (req, res) => {
  res.json({ status: 'Backend conectado correctamente' });
});

// Ruta para obtener personas
app.get('/api/personas', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT id_persona, nombre, apellido_paterno FROM personas LIMIT 10');
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al obtener personas' });
  }
});

// Ruta para crear una persona (ejemplo)
app.post('/api/personas', async (req, res) => {
  const { nombre, apellido_paterno } = req.body;
  try {
    const { rows } = await pool.query(
      'INSERT INTO personas(nombre, apellido_paterno) VALUES($1, $2) RETURNING *',
      [nombre, apellido_paterno]
    );
    res.json(rows[0]);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al crear persona' });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Servidor corriendo en puerto ${PORT}`);
});
