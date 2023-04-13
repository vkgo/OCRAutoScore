import ReactDOM from 'react-dom/client';
import './index.less';
import App from './App';
import { BrowserRouter as Router } from 'react-router-dom';
import {Provider} from 'react-redux'
import store from './store';
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <Provider store={store}>
    <Router>
       <App/> 
    </Router>
  </Provider>
);