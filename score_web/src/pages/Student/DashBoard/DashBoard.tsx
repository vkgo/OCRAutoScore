import React,{useEffect,useState} from "react"
import { renderRoutes, RouteConfigComponentProps } from "react-router-config"
import { Link } from 'react-router-dom';
import { Breadcrumb, Button, Dropdown, MenuProps, Space } from "antd"
import { DownOutlined} from '@ant-design/icons';
import BreadcrumbItemInterface from "@/ts/interface/BreadcrumbItemInterface";
import './DashBoard.less'
interface DashBoardProps extends RouteConfigComponentProps {}
const breadcrumbNameMap: Record<string, string> = {
    '/student': '首页',
    '/student/papers': '题库',
    '/student/papers/detail': '查看试卷详情',
};

const DashBoard: React.FC<DashBoardProps> = (props) => {
    const pathSnippets = props.location.pathname.split('/').filter((i) => i);
    let breadcrumbItems : BreadcrumbItemInterface[] = [];
    for(let i = 0; i < pathSnippets.length; i++) {
        const url = `/${pathSnippets.slice(0, i + 1).join('/')}`;
        if(!(url in breadcrumbNameMap)) continue;
        breadcrumbItems.push({
            key: url,
            title: url === '/student' || url==='/student/papers/detail'? breadcrumbNameMap[url] :<Link to={url}>{breadcrumbNameMap[url]}</Link>,
        });
    };

    const logOut = () => {
        window.sessionStorage.removeItem("role");
        window.sessionStorage.removeItem("username");
        props.history.push("/")
    }

    const items : MenuProps['items'] = [
        {
            key: '1',
            label: (<Button type="link" onClick={()=>logOut()}>登出</Button>)
        }
    ];
    const [username, setUsername] = useState("")

    useEffect(()=>{
        setUsername(window.sessionStorage.getItem("username"));
    })
    return (
        <div className="student_dashboard_body">
            <div className="header">
                <Breadcrumb items = {breadcrumbItems}/>
                <Dropdown menu={{items}}>
                    <Space>
                        <span>{username}同学, 你好</span>
                        <DownOutlined/>
                    </Space>
                </Dropdown>
            </div>
            {renderRoutes(props.route.routes)}
        </div>
    )
}
export default DashBoard;