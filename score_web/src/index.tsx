import ReactDOM from 'react-dom/client';
import './index.less';
import App from './App';
import { BrowserRouter as Router } from 'react-router-dom';
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
    <Router>
       <App/> 
    </Router>
);