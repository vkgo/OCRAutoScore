/**获取0~end的range数组 */
export default function range({start = 0, end, step = 1}:{start?:number, end: number,step?:number}): number[] {
    const result = [];
    for (let i = start; i < end; i += step) {
        result.push(i);
    }
    return result;
}