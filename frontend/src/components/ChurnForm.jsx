import React, { useState } from 'react';
import { Send, User, MapPin, Briefcase, CreditCard, Activity } from 'lucide-react';

const ChurnForm = ({ onPredict }) => {
    const [formData, setFormData] = useState({
        Geography: 'France',
        Gender: 'Male',
        CreditScore: 650,
        Age: 40,
        Tenure: 5,
        Balance: 50000,
        NumOfProducts: 2,
        HasCrCard: 1,
        IsActiveMember: 1,
        EstimatedSalary: 60000
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'].includes(name)
                ? parseFloat(value)
                : value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onPredict(formData);
    };

    return (
        <div className="glass-card">
            <h2 className="section-title"><User size={24} color="#6366f1" /> Individual Prediction</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-grid">
                    <div className="form-group">
                        <label>Geography</label>
                        <select name="Geography" value={formData.Geography} onChange={handleChange}>
                            <option value="France">France</option>
                            <option value="Germany">Germany</option>
                            <option value="Spain">Spain</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Gender</label>
                        <select name="Gender" value={formData.Gender} onChange={handleChange}>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Credit Score</label>
                        <input type="number" name="CreditScore" value={formData.CreditScore} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Age</label>
                        <input type="number" name="Age" value={formData.Age} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Tenure (Years)</label>
                        <input type="number" name="Tenure" value={formData.Tenure} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Balance ($)</label>
                        <input type="number" name="Balance" value={formData.Balance} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Number of Products</label>
                        <input type="number" name="NumOfProducts" value={formData.NumOfProducts} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Estimated Salary ($)</label>
                        <input type="number" name="EstimatedSalary" value={formData.EstimatedSalary} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Has Credit Card?</label>
                        <select name="HasCrCard" value={formData.HasCrCard} onChange={handleChange}>
                            <option value={1}>Yes</option>
                            <option value={0}>No</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Is Active Member?</label>
                        <select name="IsActiveMember" value={formData.IsActiveMember} onChange={handleChange}>
                            <option value={1}>Yes</option>
                            <option value={0}>No</option>
                        </select>
                    </div>
                    <button type="submit" className="btn-primary">
                        <Send size={18} /> Run Prediction
                    </button>
                </div>
            </form>
        </div>
    );
};

export default ChurnForm;
