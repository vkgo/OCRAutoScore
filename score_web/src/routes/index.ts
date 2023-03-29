import {RouteConfig} from  'react-router-config'
import Login from '@/pages/Login/Login'
import TeacherDashBoard from '@/pages/Teacher/DashBoard/DashBoard'
import AddPaperBoard from '@/pages/Teacher/AddPaperBoard/AddPaperBoard'
import PaperBoard from '@/pages/Teacher/PaperBoard/PaperBoard'
import PaperDetail from '@/pages/Teacher/PaperDetail/PaperDetail'
const routes:RouteConfig = [
    {
        path: '/login',
        exact: true,
        component: Login,
    },
    {
        path: '/teacher',
        component:TeacherDashBoard,
        routes: [
            {
                path: '/teacher/add',
                component: AddPaperBoard,
            },
            {
                path: '/teacher/list', 
                component: PaperBoard,
            },
            {
                path: '/teacher/detail/:id',
                component: PaperDetail
            }
        ]
    }
]

export default routes