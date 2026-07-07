-- DataContract Lab — database schema
-- This is a reference copy of the schema. The app creates this automatically
-- on startup (see get_engine() in app.py) — you do not need to run this file
-- by hand. It's kept here purely as documentation.

CREATE DATABASE IF NOT EXISTS datacontract_lab;

USE datacontract_lab;

CREATE TABLE IF NOT EXISTS scans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    baseline_filename VARCHAR(255) NOT NULL,
    new_filename VARCHAR(255) NOT NULL,
    quality_score INT NOT NULL,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
