import React from 'react';
import {Table, Tag} from 'antd';
import type { ColumnsType } from 'antd/es/table';

interface ScoreDataType {
    key: string;
    name: string;
    email: string;
    // tags: 优秀 意味着85分以上， 良好 在75分到84分之间， 一般在65分到74分之间 合格 在60分到64分之间
    score: number;
}

const column: ColumnsType<ScoreDataType> = [
    {
        title: 'Name',
        dataIndex: 'name',
        key: 'name',
        render: (text) => <span>{text}</span>
    },
    {
        title: 'Email',
        dataIndex: 'email',
        key: 'email',
        render: (text) => <span>{text}</span>
    },
    {
        title: 'Tag',
        dataIndex: 'score',
        key: 'score',
        render: (score) => {
            let color =  '';
            let text = '';
            if( score < 60 ) {
                color = 'red';
                text = '不合格'
            }
            else if (score <= 64 ) {
                color = 'blue';
                text = '合格';
            }
            else if(score <= 74) {
                color = '#ebeee8';
                text = '一般';
            }
            else if (score <=84){
                color = '#a6559d';
                text = '良好';
            }
            else {
                color = 'green';
                text = '优秀';
            }
            return (
                <Tag color={color}>
                    {text}
                </Tag>
            )
        }
    },
    {
        title: 'Score',
        dataIndex: 'score',
        key: 'score',
        render: (text) => <span>{text}</span>
    },
]

const tableData: ScoreDataType[] = [
    {
        key: '1',
        name:'郝涛',
        email: 'u.vverzum@dmlqmgwsh.ck',
        score: 37
    },
    {
        key: '2',
        name: '邓勇',
        email: 'q.ylkcezyrwd@cwrqnpmep.nz',
        score: 78
    }
]

const PaperDetail:React.FC = () => {
    return (
        <div>
            <h2>河北省石家庄市赵县2022-2023学年七年级下学期3月月考试题</h2>
            <p>考试成绩</p>
            <Table columns={column} dataSource={tableData}/>
        </div>
    )
}

export default PaperDetail;