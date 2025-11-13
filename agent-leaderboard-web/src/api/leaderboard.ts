import axios from 'axios';
import { LeaderboardResult } from '../types';

const API_URL = 'https://api.example.com/leaderboard'; // Replace with your actual API endpoint

export const fetchLeaderboardData = async (): Promise<LeaderboardResult[]> => {
    try {
        const response = await axios.get(API_URL);
        return response.data;
    } catch (error) {
        console.error('Error fetching leaderboard data:', error);
        throw error;
    }
};