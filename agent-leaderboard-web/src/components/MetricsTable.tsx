import React from 'react';
import styles from '../styles/leaderboard.module.css';

interface Metric {
  name: string;
  value: number;
}

interface MetricsTableProps {
  metrics: Metric[];
}

const MetricsTable: React.FC<MetricsTableProps> = ({ metrics }) => {
  return (
    <div className={styles.metricsTable}>
      <h2>Agent Metrics</h2>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric, index) => (
            <tr key={index}>
              <td>{metric.name}</td>
              <td>{metric.value.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MetricsTable;