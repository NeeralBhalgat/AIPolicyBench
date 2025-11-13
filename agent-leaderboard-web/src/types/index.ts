export interface Agent {
    id: string;
    name: string;
    hallucinationRate: number;
    performanceMetrics: Record<string, number>;
}

export interface LeaderboardResult {
    agents: Agent[];
    totalAgents: number;
    averageHallucinationRate: number;
}