import React from 'react';
import OriginalNavbar from '@theme-original/Navbar';
import type Props from '@theme/Navbar';

export default function Navbar(props: any){
  return (
    <>
      <OriginalNavbar {...props} />
    </>
  );
}