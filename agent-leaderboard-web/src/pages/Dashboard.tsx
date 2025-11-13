import React from 'react';
import Leaderboard from '../components/Leaderboard';
import Filters from '../components/Filters';
import Layout from '../components/Layout';
import { useLeaderboard } from '../hooks/useLeaderboard';

const Dashboard: React.FC = () => {
    const { data, loading, error } = useLeaderboard();

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error loading leaderboard data.</div>;
    }

    return (
        <Layout>
            <h1 className="text-2xl font-bold mb-4">Agent Leaderboard</h1>
            <Filters />
            <Leaderboard data={data} />
        </Layout>
    );
};

export default Dashboard;