CREATE TABLE IF NOT EXISTS provenance_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stage VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_source VARCHAR(255),
    output_target VARCHAR(255),
    records_in INT,
    records_out INT,
    duration_seconds FLOAT,
    extra JSON
);
