import {RouteConfig} from  'react-router-config'
import Login from '@/pages/Login/Login'

const routes:RouteConfig = [
    {
        path: '/login',
        exact: true,
        component: Login,
    }
]

export default routes