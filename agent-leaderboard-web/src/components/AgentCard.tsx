import React from 'react';
import { Agent } from '../types';

interface AgentCardProps {
  agent: Agent;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent }) => {
  return (
    <div className="agent-card">
      <h3 className="agent-name">{agent.name}</h3>
      <p className="hallucination-rate">Hallucination Rate: {agent.hallucinationRate}%</p>
      <p className="performance-metric">Performance Metric: {agent.performanceMetric}</p>
      <p className="last-updated">Last Updated: {new Date(agent.lastUpdated).toLocaleDateString()}</p>
    </div>
  );
};

export default AgentCard;