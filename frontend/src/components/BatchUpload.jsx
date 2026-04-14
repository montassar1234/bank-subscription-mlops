import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import axios from 'axios';

const BatchUpload = () => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setError('');
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        setError('');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:8000/predict_batch', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResults(response.data);
        } catch (err) {
            setError('Failed to process batch. Ensure CSV columns match requirements.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="glass-card">
            <h2 className="section-title"><Upload size={24} color="#6366f1" /> Batch Prediction</h2>

            <div className="dropzone" onClick={() => document.getElementById('csvInput').click()}>
                <input
                    id="csvInput"
                    type="file"
                    accept=".csv"
                    hidden
                    onChange={handleFileChange}
                />
                {file ? (
                    <div>
                        <CheckCircle size={40} color="#22c55e" style={{ margin: '0 auto' }} />
                        <p>{file.name}</p>
                    </div>
                ) : (
                    <div>
                        <FileText size={40} color="#94a3b8" style={{ margin: '0 auto' }} />
                        <p>Click to select CSV file</p>
                    </div>
                )}
            </div>

            <button
                className="btn-primary"
                onClick={handleUpload}
                disabled={!file || loading}
                style={{ width: '100%', opacity: (!file || loading) ? 0.5 : 1 }}
            >
                {loading ? 'Processing...' : 'Upload & Process'}
            </button>

            {error && <p style={{ color: '#ef4444', marginTop: '1rem', fontSize: '0.875rem' }}><AlertCircle size={14} inline /> {error}</p>}

            {results.length > 0 && (
                <div className="batch-table-container">
                    <table className="batch-table">
                        <thead>
                            <tr>
                                <th>Customer</th>
                                <th>Score</th>
                                <th>Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results.slice(0, 5).map((row, idx) => (
                                <tr key={idx}>
                                    <td>Customer {idx + 1}</td>
                                    <td>{(row.Churn_Probability * 100).toFixed(1)}%</td>
                                    <td>
                                        <span className={row.Churn_Prediction === 1 ? 'status-exited' : 'status-stayed'}>
                                            {row.Churn_Prediction === 1 ? 'Exit' : 'Stay'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>Showing first 5 of {results.length} rows.</p>
                </div>
            )}
        </div>
    );
};

export default BatchUpload;
