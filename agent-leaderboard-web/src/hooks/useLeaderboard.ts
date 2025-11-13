import { useEffect, useState } from 'react';
import { fetchLeaderboardData } from '../api/leaderboard';
import { LeaderboardResult } from '../types';

const useLeaderboard = () => {
    const [data, setData] = useState<LeaderboardResult[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const loadData = async () => {
            try {
                setLoading(true);
                const result = await fetchLeaderboardData();
                setData(result);
            } catch (err) {
                setError('Failed to load leaderboard data');
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, []);

    return { data, loading, error };
};

export default useLeaderboard;