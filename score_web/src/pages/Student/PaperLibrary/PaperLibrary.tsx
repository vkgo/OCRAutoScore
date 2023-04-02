import PaperList from '@/components/PaperList/PaperList';
import React from 'react';

const PaperLibrary: React.FC = () => {
    return (
        <div>
            <h2>题库</h2>
            <PaperList baseUrl='/student/papers/detail/'/>
        </div>
    )
}

export default PaperLibrary;