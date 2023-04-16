import React,{useEffect, useState} from "react"
import { RouteConfigComponentProps, renderRoutes } from "react-router-config"
import { Link } from 'react-router-dom';
import './DashBoard.less'
import { Breadcrumb, Button, Dropdown, MenuProps, Space } from "antd"
import { DownOutlined} from '@ant-design/icons';
import BreadcrumbItemInterface from "@/ts/interface/BreadcrumbItemInterface";
interface DashBoardProps extends RouteConfigComponentProps {}
const breadcrumbNameMap: Record<string, string> = {
    '/teacher': '首页',
    '/teacher/list': '查看试卷列表',
    '/teacher/list/add': '添加试卷',
    '/teacher/list/detail': '查看试卷详情',
};

const DashBoard: React.FC<DashBoardProps> = (props) => {
    console.log('props:', props)
    // 拆分当前路径
    const pathSnippets = props.location.pathname.split('/').filter((i) => i);
    console.log('pathSnippets:', pathSnippets)
    let breadcrumbItems : BreadcrumbItemInterface[] = [];
    for(let i = 0; i < pathSnippets.length; i++) {
        const url = `/${pathSnippets.slice(0, i + 1).join('/')}`;
        if(!(url in breadcrumbNameMap)) continue;
        breadcrumbItems.push({
            key: url,
            title: url === '/teacher' || url==='/teacher/list/detail'? breadcrumbNameMap[url] :<Link to={url}>{breadcrumbNameMap[url]}</Link>,
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
        <div className="techer_dashboard_body">
            <div className="header">
                <Breadcrumb items = {breadcrumbItems}/>
                <Dropdown menu={{items}}>
                    <Space>
                        <span>{username}老师, 你好</span>
                        <DownOutlined/>
                    </Space>
                </Dropdown>
            </div>
            {renderRoutes(props.route.routes)}
        </div>
    )
}
export default DashBoard;