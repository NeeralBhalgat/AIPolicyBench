import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import AgentDetail from './pages/AgentDetail';
import Layout from './components/Layout';

const App: React.FC = () => {
    return (
        <Router>
            <Layout>
                <Switch>
                    <Route path="/" exact component={Dashboard} />
                    <Route path="/agent/:id" component={AgentDetail} />
                </Switch>
            </Layout>
        </Router>
    );
};

export default App;