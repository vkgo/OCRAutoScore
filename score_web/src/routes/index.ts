import {RouteConfig} from  'react-router-config'
import Login from '@/pages/Login/Login'
import TeacherDashBoard from '@/pages/Teacher/DashBoard/DashBoard'
import AddPaperBoard from '@/pages/Teacher/AddPaperBoard/AddPaperBoard'
import PaperBoard from '@/pages/Teacher/PaperBoard/PaperBoard'
import TeacherPaperDetail from '@/pages/Teacher/PaperDetail/PaperDetail'
import StudentDashBoard from '@/pages/Student/DashBoard/DashBoard'
import PaperLibrary from '@/pages/Student/PaperLibrary/PaperLibrary';
import StudentPaperDetail from '@/pages/Student/PaperDetail/PaperDetail';
const routes:RouteConfig = [
    {
        path: '/',
        exact: true,
        component: Login,
    },
    {
        path: '/teacher',
        component:TeacherDashBoard,
        routes: [
            {
                path: '/teacher/list', 
                component: PaperBoard,
                exact: true
            },
            {
                path: '/teacher/list/add',
                component: AddPaperBoard,
            },
            {
                path: '/teacher/list/detail/:id',
                component: TeacherPaperDetail
            }
        ]
    },
    {
        path: '/student',
        component: StudentDashBoard,
        routes: [
            {
                path: '/student/papers',
                component: PaperLibrary,
                exact: true
            },
            {
                path: '/student/papers/detail/:id',
                component: StudentPaperDetail
            }
        ]
    }
]

export default routes