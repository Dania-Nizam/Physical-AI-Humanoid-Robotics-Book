import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotWidget from '../../components/Chatbot/ChatbotWidget';
import type {Props} from '@theme/Layout';

export default function Layout(props: Props) {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatbotWidget />
    </>
  );
}