import React from 'react';
import { useParams } from 'react-router-dom';
import { useLeaderboard } from '../hooks/useLeaderboard';
import MetricsTable from '../components/MetricsTable';
import LineChart from '../components/Charts/LineChart';
import BarChart from '../components/Charts/BarChart';
import styles from '../styles/leaderboard.module.css';

const AgentDetail = () => {
    const { agentId } = useParams();
    const { data, loading, error } = useLeaderboard(agentId);

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error loading agent details.</div>;
    }

    const { name, hallucinationRate, performanceMetrics } = data;

    return (
        <div className={styles.agentDetail}>
            <h1>{name}</h1>
            <h2>Hallucination Rate: {hallucinationRate}%</h2>
            <MetricsTable metrics={performanceMetrics} />
            <LineChart data={performanceMetrics} />
            <BarChart data={performanceMetrics} />
        </div>
    );
};

export default AgentDetail;