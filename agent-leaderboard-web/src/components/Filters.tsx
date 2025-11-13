import React from 'react';

const Filters: React.FC<{ onFilterChange: (filter: string) => void }> = ({ onFilterChange }) => {
    const handleFilterChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        onFilterChange(event.target.value);
    };

    return (
        <div className="filters">
            <label htmlFor="filter-select">Filter by:</label>
            <select id="filter-select" onChange={handleFilterChange}>
                <option value="all">All Agents</option>
                <option value="hallucination">Hallucination Rates</option>
                <option value="performance">Performance Metrics</option>
                <option value="recent">Recently Added</option>
            </select>
        </div>
    );
};

export default Filters;