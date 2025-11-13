import React from 'react';
import styles from '../styles/leaderboard.module.css';
import { useLeaderboard } from '../hooks/useLeaderboard';
import MetricsTable from './MetricsTable';
import Filters from './Filters';

const Leaderboard: React.FC = () => {
    const { data, loading, error } = useLeaderboard();

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error loading leaderboard data.</div>;
    }

    return (
        <div className={styles.leaderboardContainer}>
            <h1 className={styles.title}>Agent Leaderboard</h1>
            <Filters />
            <MetricsTable data={data} />
        </div>
    );
};

export default Leaderboard;