import {RouteConfig} from  'react-router-config'
import Login from '@/pages/Login/Login'
import TeacherDashBoard from '@/pages/Teacher/DashBoard/DashBoard'
const routes:RouteConfig = [
    {
        path: '/login',
        exact: true,
        component: Login,
    },
    {
        path: '/teacher',
        exact: true,
        component:TeacherDashBoard 
    }
]

export default routes