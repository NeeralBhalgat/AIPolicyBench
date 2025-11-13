# Agent Leaderboard Web

This project implements a leaderboard interface for agents, displaying their performance metrics, including hallucination rates, in an attractive web format. The application is built using React and TypeScript, and it utilizes Vite for development and build processes.

## Project Structure

The project is organized as follows:

```
agent-leaderboard-web
├── src
│   ├── main.tsx               # Entry point of the React application
│   ├── App.tsx                # Main application component with routing
│   ├── pages
│   │   ├── Dashboard.tsx      # Displays the leaderboard of agents
│   │   └── AgentDetail.tsx    # Shows detailed information about a specific agent
│   ├── components
│   │   ├── Leaderboard.tsx     # Presents the list of agents and their scores
│   │   ├── AgentCard.tsx       # Displays a summary of an individual agent's performance
│   │   ├── Filters.tsx         # Allows users to filter leaderboard results
│   │   ├── MetricsTable.tsx    # Displays detailed metrics for agents
│   │   ├── Charts
│   │   │   ├── LineChart.tsx   # Visualizes data trends over time
│   │   │   └── BarChart.tsx    # Visualizes comparative data
│   │   └── Layout.tsx          # Defines the overall structure of the application
│   ├── hooks
│   │   └── useLeaderboard.ts    # Custom hook for managing leaderboard data
│   ├── api
│   │   └── leaderboard.ts       # Functions for fetching leaderboard data
│   ├── types
│   │   └── index.ts            # TypeScript types and interfaces
│   ├── utils
│   │   └── formatters.ts       # Utility functions for formatting data
│   ├── data
│   │   └── sample-results.json  # Sample data for the leaderboard
│   └── styles
│       ├── global.css          # Global styles for the application
│       └── leaderboard.module.css # Module-specific styles for the leaderboard
├── public
│   └── index.html              # Main HTML file for the application
├── package.json                # npm configuration file
├── tsconfig.json               # TypeScript configuration file
├── vite.config.ts              # Vite configuration file
├── tailwind.config.js          # Tailwind CSS configuration file
└── README.md                   # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd agent-leaderboard-web
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000` to view the application.

## Usage Guidelines

- The **Dashboard** page displays the leaderboard with various metrics for each agent.
- Click on an agent's name to view detailed performance metrics on the **Agent Detail** page.
- Use the **Filters** component to refine the leaderboard results based on specific criteria.
- The application includes visualizations such as line and bar charts to help interpret the data effectively.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.